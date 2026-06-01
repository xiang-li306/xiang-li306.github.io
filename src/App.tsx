import { BlogList } from "./components/BlogList";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { MiscList } from "./components/MiscList";
import { Navbar } from "./components/Navbar";
import { PublicationList } from "./components/PublicationList";
import { Section } from "./components/Section";
import { blogPosts } from "./data/blog";
import { miscellaneous } from "./data/misc";
import { publications } from "./data/publications";
import { siteConfig } from "./data/site";

export default function App() {
  return (
    <div className="min-h-screen bg-white text-ink-900">
      <Navbar links={siteConfig.navigation} />
      <main className="mx-auto w-full max-w-5xl px-5 pb-16 pt-10 sm:px-8 sm:pt-14">
        <Hero profile={siteConfig.profile} />
        <Section
          id="publications"
          title="Publications"
          description="Selected papers and preprints. Replace these placeholders with your own publication records as they become available."
        >
          <PublicationList publications={publications} />
        </Section>
        <Section
          id="notes"
          title="Blog / Notes"
          description="Short-form writing, research notes, reading notes, and occasional essays."
        >
          <BlogList posts={blogPosts} />
        </Section>
        <Section
          id="misc"
          title="Miscellaneous"
          description="Teaching, service, talks, side projects, and other academic or personal items."
        >
          <MiscList items={miscellaneous} />
        </Section>
      </main>
      <Footer profile={siteConfig.profile} />
    </div>
  );
}
