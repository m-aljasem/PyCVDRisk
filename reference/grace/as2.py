"""Minimal ActionScript 2 disassembler: one text line per action, constant-pool names resolved.

Only what reference/grace/calculator.py needs: every action block of the file is found by its ConstantPool
header (frame scripts, clip-event handlers and button handlers alike) and listed in file order.
"""
import struct
import sys

NAMES = {0x00: "End", 0x04: "NextFrame", 0x06: "Play", 0x07: "Stop", 0x0A: "Add", 0x0B: "Subtract", 0x0C: "Multiply",
         0x0D: "Divide", 0x0E: "Equals", 0x0F: "Less", 0x10: "And", 0x11: "Or", 0x12: "Not", 0x13: "StringEquals",
         0x14: "StringLength", 0x17: "Pop", 0x18: "ToInteger", 0x1C: "GetVariable", 0x1D: "SetVariable",
         0x20: "SetTarget2", 0x21: "StringAdd", 0x22: "GetProperty", 0x23: "SetProperty", 0x26: "Trace",
         0x3A: "Delete", 0x3B: "Delete2", 0x3C: "DefineLocal", 0x3D: "CallFunction", 0x3E: "Return", 0x3F: "Modulo",
         0x40: "NewObject", 0x41: "DefineLocal2", 0x42: "InitArray", 0x43: "InitObject", 0x44: "TypeOf",
         0x46: "Enumerate", 0x47: "Add2", 0x48: "Less2", 0x49: "Equals2", 0x4A: "ToNumber", 0x4B: "ToString",
         0x4C: "PushDuplicate", 0x4D: "StackSwap", 0x4E: "GetMember", 0x4F: "SetMember", 0x50: "Increment",
         0x51: "Decrement", 0x52: "CallMethod", 0x53: "NewMethod", 0x54: "InstanceOf", 0x55: "Enumerate2",
         0x60: "BitAnd", 0x61: "BitOr", 0x62: "BitXor", 0x63: "BitLShift", 0x64: "BitRShift", 0x65: "BitURShift",
         0x66: "StrictEquals", 0x67: "Greater", 0x68: "StringGreater", 0x69: "Extends", 0x81: "GotoFrame",
         0x83: "GetURL", 0x87: "StoreRegister", 0x88: "ConstantPool", 0x8A: "WaitForFrame", 0x8B: "SetTarget",
         0x8C: "GoToLabel", 0x8E: "DefineFunction2", 0x94: "With", 0x96: "Push", 0x99: "Jump", 0x9A: "GetURL2",
         0x9B: "DefineFunction", 0x9D: "If", 0x9F: "GotoFrame2", 0x2A: "Throw", 0x8F: "Try", 0x2B: "CastOp",
         0x2C: "ImplementsOp"}


def cstr(b, i):
    j = b.index(0, i)
    return b[i:j].decode("latin-1"), j + 1


def actions(b, i, end, out, depth=0, pool=None):
    pool = [] if pool is None else pool
    while i < end:
        code = b[i]
        i += 1
        if code == 0:
            out.append("  " * depth + "End")
            continue
        n = 0
        if code >= 0x80:
            n = struct.unpack_from("<H", b, i)[0]
            i += 2
        d = b[i:i + n]
        name = NAMES.get(code, f"op{code:02x}")
        arg = ""
        if code == 0x88:
            cnt = struct.unpack_from("<H", d, 0)[0]
            k = 2
            pool.clear()
            for _ in range(cnt):
                s, k = cstr(d, k)
                pool.append(s)
            arg = f"{len(pool)} strings"
            out.append(("POOL", pool))
        elif code == 0x96:
            k, vals = 0, []
            while k < n:
                t = d[k]
                k += 1
                if t == 0:
                    s, k = cstr(d, k); vals.append(repr(s))
                elif t == 1:
                    vals.append(repr(struct.unpack_from("<f", d, k)[0])); k += 4
                elif t in (2, 3):
                    vals.append("null" if t == 2 else "undef")
                elif t == 4:
                    vals.append(f"r{d[k]}"); k += 1
                elif t == 5:
                    vals.append(str(bool(d[k]))); k += 1
                elif t == 6:
                    hi, lo = d[k:k + 4], d[k + 4:k + 8]
                    vals.append(repr(struct.unpack("<d", lo + hi)[0])); k += 8
                elif t == 7:
                    vals.append(str(struct.unpack_from("<i", d, k)[0])); k += 4
                elif t == 8:
                    vals.append(repr(pool[d[k]]) if d[k] < len(pool) else f"c{d[k]}"); k += 1
                elif t == 9:
                    c = struct.unpack_from("<H", d, k)[0]; vals.append(repr(pool[c]) if c < len(pool) else f"c{c}"); k += 2
                else:
                    vals.append(f"?type{t}"); break
            arg = " ".join(vals)
        elif code in (0x99, 0x9D):
            arg = str(struct.unpack_from("<h", d, 0)[0])
        elif code == 0x87:
            arg = f"r{d[0]}"
        elif code in (0x9B, 0x8E):
            fname, k = cstr(d, 0)
            np_ = struct.unpack_from("<H", d, k)[0]; k += 2
            ps = []
            if code == 0x8E:
                k += 3
                for _ in range(np_):
                    reg = d[k]; s, k = cstr(d, k + 1); ps.append(f"{s}@r{reg}" if reg else s)
            else:
                for _ in range(np_):
                    s, k = cstr(d, k); ps.append(s)
            size = struct.unpack_from("<H", d, k)[0]
            out.append("  " * depth + f"{name} {fname}({', '.join(ps)}) size={size}")
            i += n
            actions(b, i, i + size, out, depth + 1, pool)
            i += size
            continue
        elif code == 0x8C or code == 0x83:
            arg = d.decode("latin-1", "replace")
        out.append("  " * depth + f"{name} {arg}".rstrip())
        i += n
    return i


def disassemble(b):
    """Text lines of every action block in an uncompressed SWF body (a ConstantPool whose declared
    length matches its strings starts a block)."""
    lines = []
    i = 0
    while True:
        i = b.find(b"\x88", i)
        if i < 0:
            return lines
        try:
            n, cnt = struct.unpack_from("<HH", b, i + 1)
            k, ok = i + 5, True
            for _ in range(cnt):
                k = b.index(0, k) + 1
            ok = cnt > 0 and k == i + 3 + n
        except (ValueError, struct.error):
            ok = False
        if ok:
            out = []
            try:
                actions_until_end(b, i, out)
            except Exception as e:  # noqa: BLE001
                out.append(f"!! {e}")
            lines.append(f"### block at {i}")
            lines += [x for x in out if not isinstance(x, tuple)]
            i += 1
        else:
            i += 1


def actions_until_end(b, i, out):
    # find the End (0x00) that closes this block at top level by walking records
    j = i
    while b[j] != 0:
        j += 1 + (2 + struct.unpack_from("<H", b, j + 1)[0] if b[j] >= 0x80 else 0)
    actions(b, i, j + 1, out)

