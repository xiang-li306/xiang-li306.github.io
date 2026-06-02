import type { NavigationLink } from "../data/site";

type NavbarProps = {
  links: NavigationLink[];
  title: string;
};

export function Navbar({ links, title }: NavbarProps) {
  return (
    <header className="sticky top-0 z-20 border-b border-ink-200 bg-white/95 backdrop-blur">
      <nav className="mx-auto flex w-full max-w-5xl flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <a
          href="#about"
          className="text-base font-semibold tracking-tight text-ink-900 hover:text-ink-700"
        >
          {title}
        </a>
        <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-ink-700">
          {links.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="hover:text-ink-900 hover:underline"
            >
              {link.label}
            </a>
          ))}
        </div>
      </nav>
    </header>
  );
}
