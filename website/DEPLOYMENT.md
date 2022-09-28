# Deployment Guide for Vercel

This guide will help you deploy the CVD Risk Calculator website to Vercel.

## Prerequisites

1. A GitHub account
2. A Vercel account (free tier works)
3. Your code pushed to GitHub

## Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Go to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Sign in with your GitHub account

2. **Import Your Project**
   - Click "Add New Project"
   - Select your GitHub repository
   - Choose the repository containing the `website` folder

3. **Configure Project**
   - **Root Directory**: Set to `website`
   - **Framework Preset**: Next.js (auto-detected)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
   - **Install Command**: `npm install` (default)

4. **Environment Variables**
   - Add any environment variables if needed
   - For now, none are required

5. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - Your site will be live!

### Option 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to website directory
cd website

# Login to Vercel
vercel login

# Deploy
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? (Choose your account)
# - Link to existing project? No
# - What's your project's name? cvd-risk-calculator
# - In which directory is your code located? ./
```

### Option 3: GitHub Integration

1. **Enable GitHub Integration**
   - In Vercel dashboard, go to Settings > Git
   - Connect your GitHub account if not already connected

2. **Auto Deployments**
   - Every push to `main` branch will trigger a deployment
   - Pull requests get preview deployments automatically

## Configuration

The website is configured for static export (see `next.config.js`):
- Uses `output: 'export'` for static site generation
- Optimized for Vercel's platform

## Custom Domain

1. Go to your project in Vercel dashboard
2. Navigate to Settings > Domains
3. Add your custom domain
4. Follow DNS configuration instructions

## Build Configuration

The project uses:
- **Framework**: Next.js 14
- **Build Command**: `npm run build`
- **Output Directory**: `out` (static export)
- **Node Version**: 18.x or higher (auto-detected)

## Troubleshooting

### Build Fails

1. Check build logs in Vercel dashboard
2. Ensure all dependencies are in `package.json`
3. Verify Node version (should be 18+)

### Site Not Updating

1. Check if GitHub integration is enabled
2. Verify you're pushing to the correct branch
3. Check Vercel dashboard for deployment status

### Styling Issues

1. Ensure Tailwind CSS is properly configured
2. Check that `styles/globals.css` is imported in `_app.tsx`

## Performance

- The site uses Next.js static export for optimal performance
- Images should be optimized (add to `public/` folder)
- Consider adding analytics (Google Analytics, Vercel Analytics)

## Next Steps

After deployment:
1. Update GitHub links in the website code
2. Add real API endpoints if implementing backend
3. Add analytics
4. Set up custom domain
5. Configure SEO metadata

