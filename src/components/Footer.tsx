type FooterProps = {
  profile: {
    name: string;
    email: string;
  };
};

export function Footer({ profile }: FooterProps) {
  return (
    <footer className="border-t border-ink-200">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-2 px-5 py-8 text-sm text-ink-500 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p>&copy; {new Date().getFullYear()} {profile.name}</p>
        <a href={`mailto:${profile.email}`} className="hover:text-ink-900 hover:underline">
          {profile.email}
        </a>
      </div>
    </footer>
  );
}
