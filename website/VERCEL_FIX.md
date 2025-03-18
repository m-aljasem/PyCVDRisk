# Vercel Deployment Fix

## The Issue
Vercel is looking for files in `/out` directory, but Next.js outputs to `.next` directory by default.

## Solution Steps

### Option 1: Delete vercel.json and let Vercel auto-detect (RECOMMENDED)

1. **Delete `vercel.json` file** if it exists
2. **In Vercel Dashboard:**
   - Go to your project settings
   - Delete the project (or create a new one)
   - Re-import from GitHub
   - **DO NOT** create `vercel.json` manually
   - Let Vercel auto-detect Next.js

### Option 2: Manually Configure in Vercel Dashboard

1. Go to **Project Settings** → **General**
2. Make sure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `website`
   - **Build Command**: `npm run build` (should be auto-detected)
   - **Output Directory**: Leave EMPTY or set to `.next` (NOT `out`)

3. Go to **Project Settings** → **Build & Development Settings**
   - **Override**: Turn OFF any overrides
   - Let Vercel use its defaults

### Option 3: Clear Vercel Cache

If you previously deployed with wrong settings:

1. Go to **Project Settings** → **General**
2. **Delete Project** (or disconnect from GitHub)
3. Create a **new project** and connect GitHub
4. Set root directory to `website`
5. Deploy fresh

### Current Configuration

✅ `next.config.js` - Correct (no `output: 'export'`)
✅ `package.json` - Standard Next.js scripts
✅ No `vercel.json` - Let Vercel auto-detect

### Verification Commands

Test locally first:
```bash
cd website
npm install
npm run build
```

You should see:
```
Creating an optimized production build...
✓ Compiled successfully
○ Static pages: X
○ Dynamic pages: Y
```

The build should create `.next` folder, NOT `out` folder.

### If Still Not Working

1. **Check if there's a cached `.vercel` folder:**
   ```bash
   rm -rf .vercel
   ```

2. **Remove any `out` directory:**
   ```bash
   rm -rf out
   ```

3. **Check package.json scripts:**
   ```json
   "build": "next build"  // Should be this, NOT "next build && next export"
   ```

4. **Redeploy in Vercel:**
   - Create fresh project
   - Don't override any settings
   - Let Vercel auto-detect everything

