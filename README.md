# Xiang Li Academic Homepage

This repository contains the source for [xiang-li306.github.io](https://xiang-li306.github.io), built with Vite, React, TypeScript, and Tailwind CSS.

The site is structured for long-term maintenance: editable academic content lives in `src/data`, reusable view pieces live in `src/components`, and static files such as a profile photo or CV belong in `public`.

## Local Development

Install dependencies:

```bash
npm install
```

Start the local development server:

```bash
npm run dev
```

Vite will print a local URL, usually `http://localhost:5173`.

## Build

Create a production build:

```bash
npm run build
```

The built site will be written to `dist`.

## Preview the Production Build

Preview the built site locally:

```bash
npm run preview
```

Vite will print a local preview URL, usually `http://localhost:4173`.

## Editing Content

Most routine updates should only require editing files in `src/data`:

- `src/data/site.ts` for name, affiliation, research interests, links, email, and profile photo.
- `src/data/publications.ts` for papers, preprints, drafts, and links.
- `src/data/blog.ts` for blog posts, research notes, essays, or reading notes.
- `src/data/misc.ts` for teaching, service, talks, and other items.

To replace the placeholder profile image, add your image to `public`, then update `photo` in `src/data/site.ts`. For example, if you add `public/profile.jpg`, set `photo: "/profile.jpg"`.

To add a CV, place `cv.pdf` in `public` so the existing `/cv.pdf` link works.

## Deployment

This repository is configured for GitHub Pages with Vite `base: "/"`, which is correct for a GitHub user page.

The included GitHub Actions workflow builds the site and deploys `dist` to GitHub Pages whenever changes are pushed to the `main` branch.

After the first push:

1. Open the repository on GitHub.
2. Go to **Settings > Pages**.
3. Under **Build and deployment**, choose **GitHub Actions** as the source if it is not already selected.
4. Wait for the deployment workflow to finish.
5. Visit [https://xiang-li306.github.io](https://xiang-li306.github.io).
