import type { ProfileLink, TextSegment } from "../data/site";

type HeroProps = {
  profile: {
    name: string;
    displayName: string;
    affiliation: {
      name: string;
      href: string;
    };
    email: string;
    photo: string;
    intro: TextSegment[];
    researchSummary: string;
    interests: string[];
    links: ProfileLink[];
  };
};

export function Hero({ profile }: HeroProps) {
  return (
    <section
      id="about"
      className="hero-section grid gap-9 border-b border-ink-200 pb-12"
    >
      <div>
        <h1 className="text-4xl font-semibold tracking-tight text-ink-900 sm:text-5xl">
          {profile.name}
        </h1>
        <div className="mt-4 space-y-1 text-base leading-7 text-ink-700">
          <p>
            <a
              href={profile.affiliation.href}
              className="underline decoration-ink-200 hover:text-ink-900 hover:decoration-ink-700"
            >
              {profile.affiliation.name}
            </a>
          </p>
        </div>
        <p className="mt-6 max-w-3xl text-lg leading-8 text-ink-700">
          {profile.intro.map((segment) =>
            segment.href ? (
              <a
                key={`${segment.text}-${segment.href}`}
                href={segment.href}
                className="underline decoration-ink-200 hover:text-ink-900 hover:decoration-ink-700"
              >
                {segment.text}
              </a>
            ) : (
              segment.text
            ),
          )}
        </p>
        <p className="mt-4 max-w-3xl text-lg leading-8 text-ink-700">
          {profile.researchSummary}
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          {profile.interests.map((interest) => (
            <span
              key={interest}
              className="rounded border border-ink-200 px-2.5 py-1 text-sm text-ink-700"
            >
              {interest}
            </span>
          ))}
        </div>
        <div className="mt-7 flex flex-wrap gap-x-5 gap-y-2 text-sm font-medium">
          {profile.links.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="text-ink-900 underline decoration-ink-200 hover:decoration-ink-700"
            >
              {link.label}
            </a>
          ))}
        </div>
      </div>
      <div className="hero-photo w-44">
        <img
          src={profile.photo}
          alt={`${profile.name} profile`}
          className="aspect-square w-full rounded-full border border-ink-200 object-cover"
        />
      </div>
    </section>
  );
}
