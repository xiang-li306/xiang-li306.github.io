import { BlogList } from "./components/BlogList";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { NotePage } from "./components/NotePage";
import { MiscList } from "./components/MiscList";
import { Navbar } from "./components/Navbar";
import { PublicationList } from "./components/PublicationList";
import { Section } from "./components/Section";
import { blogPosts } from "./data/blog";
import { miscellaneous } from "./data/misc";
import { publications } from "./data/publications";
import { siteConfig } from "./data/site";
import { useHashRoute } from "./hooks/useHashRoute";

export default function App() {
  const route = useHashRoute();
  const selectedNote = route.startsWith("notes/")
    ? blogPosts.find((post) => post.slug === route.replace("notes/", ""))
    : undefined;

  return (
    <div className="min-h-screen bg-white text-ink-900">
      <Navbar links={siteConfig.navigation} title={siteConfig.profile.displayName} />
      <main className="mx-auto w-full max-w-5xl px-5 pb-16 pt-10 sm:px-8 sm:pt-14">
        {selectedNote?.type === "markdown" ? (
          <NotePage post={selectedNote} />
        ) : (
          <>
            <Hero profile={siteConfig.profile} />
            <Section
              id="publications"
              title="Publications"
            >
              <PublicationList publications={publications} />
            </Section>
            <Section
              id="notes"
              title="Blog / Notes"
            >
              <BlogList posts={blogPosts} />
            </Section>
            <Section
              id="misc"
              title="Miscellaneous"
            >
              <MiscList items={miscellaneous} />
            </Section>
          </>
        )}
      </main>
      <Footer profile={siteConfig.profile} />
    </div>
  );
}
