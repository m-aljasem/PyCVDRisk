# Deployment Guide for Vercel

This guide will help you deploy the CVD Risk Calculator website to Vercel.

## ✅ Fixed Configuration

The website is now configured to work properly with Vercel's Next.js deployment.

## Quick Deploy

### Option 1: Via Vercel Dashboard (Recommended)

1. **Go to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Sign in with your GitHub account

2. **Import Your Project**
   - Click "Add New Project"
   - Select your GitHub repository
   - **IMPORTANT**: In project settings, set **Root Directory** to `website`
   
3. **Vercel Auto-Detection**
   - Vercel will automatically detect Next.js
   - Framework Preset: Next.js (auto-detected)
   - Build Command: `npm run build` (default)
   - Output Directory: `.next` (auto-detected)
   - Install Command: `npm install` (default)

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - Your site will be live! 🎉

### Option 2: Via Vercel CLI

```bash
# Install Vercel CLI (globally)
npm i -g vercel

# Navigate to website directory
cd website

# Login to Vercel
vercel login

# Deploy (for production)
vercel --prod

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? (Choose your account)
# - Link to existing project? No
# - What's your project's name? cvd-risk
# - In which directory is your code located? ./
```

## Important Settings in Vercel Dashboard

When importing the project:

1. **Root Directory**: Set to `website`
   - This tells Vercel where your Next.js app is located

2. **Framework Preset**: Should auto-detect as "Next.js"
   - If not, manually select "Next.js"

3. **Build Settings**: Leave as default
   - Build Command: `npm run build`
   - Output Directory: `.next` (auto-detected by Vercel)
   - Install Command: `npm install`

## Configuration Files

The project uses:
- **next.config.js** - Standard Next.js config (no static export)
- **vercel.json** - Minimal config (Vercel auto-detects most settings)
- **package.json** - Standard npm scripts

## Troubleshooting

### Error: routes-manifest.json not found

✅ **FIXED**: This was caused by using `output: 'export'` in next.config.js. 
The config has been updated to work with Vercel's serverless functions.

### Build Fails

1. **Check Node Version**
   - Vercel uses Node 18.x by default
   - You can specify in `package.json`:
     ```json
     "engines": {
       "node": ">=18.0.0"
     }
     ```

2. **Check Build Logs**
   - View detailed logs in Vercel dashboard
   - Look for dependency or TypeScript errors

3. **Verify Dependencies**
   - Ensure all dependencies are in `package.json`
   - Run `npm install` locally to test

### Site Not Loading

1. **Check Root Directory**
   - Must be set to `website` in Vercel project settings

2. **Check Framework Detection**
   - Should show "Next.js" as framework

3. **Redeploy**
   - Sometimes a fresh deploy fixes issues

## Environment Variables

Currently, no environment variables are needed. If you add API keys or secrets later:

1. Go to Project Settings > Environment Variables
2. Add your variables
3. Redeploy

## Custom Domain

1. Go to your project in Vercel dashboard
2. Navigate to **Settings > Domains**
3. Add your custom domain
4. Follow DNS configuration instructions

## Performance

The site is optimized with:
- Next.js automatic optimizations
- Image optimization
- Automatic code splitting
- Server-side rendering where beneficial

## Continuous Deployment

By default, Vercel will:
- ✅ Deploy on every push to `main` branch
- ✅ Create preview deployments for pull requests
- ✅ Show build status in GitHub

## Next Steps After Deployment

1. ✅ Update GitHub links in the website code
2. ✅ Add real favicon
3. ✅ Configure custom domain (optional)
4. ✅ Add analytics (optional)
5. ✅ Test all links and functionality

## Need Help?

- Check Vercel documentation: https://vercel.com/docs
- View build logs in Vercel dashboard
- Check Next.js deployment guide: https://nextjs.org/docs/deployment
