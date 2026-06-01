import type { ProfileLink } from "../data/site";

type HeroProps = {
  profile: {
    name: string;
    role: string;
    affiliation: string;
    location: string;
    email: string;
    photo: string;
    researchSummary: string;
    interests: string[];
    links: ProfileLink[];
  };
};

export function Hero({ profile }: HeroProps) {
  return (
    <section
      id="about"
      className="grid gap-9 border-b border-ink-200 pb-12 md:grid-cols-[minmax(0,1fr)_220px] md:items-start"
    >
      <div>
        <p className="mb-3 text-sm font-medium uppercase tracking-[0.16em] text-ink-500">
          Academic homepage
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-ink-900 sm:text-5xl">
          {profile.name}
        </h1>
        <div className="mt-4 space-y-1 text-base leading-7 text-ink-700">
          <p>{profile.role}</p>
          <p>{profile.affiliation}</p>
          <p>{profile.location}</p>
        </div>
        <p className="mt-6 max-w-3xl text-lg leading-8 text-ink-700">
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
      <div className="w-44 md:ml-auto md:w-full">
        <img
          src={profile.photo}
          alt={`${profile.name} profile placeholder`}
          className="aspect-[4/5] w-full rounded border border-ink-200 object-cover"
        />
      </div>
    </section>
  );
}
