import type { MouseEvent } from "react";

type MarkdownViewProps = {
  content: string;
};

function slugifyHeading(text: string) {
  return text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

function escapeHtml(text: string) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inlineMarkdown(text: string) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2" />')
    .replace(/\[([^\]]+)\]\((#[^)]+)\)/g, '<a href="$2" data-note-anchor="$2">$1</a>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function renderMarkdown(content: string) {
  const lines = content.split(/\r?\n/);
  const html: string[] = [];
  let paragraph: string[] = [];
  let listStack: Array<"ul" | "ol"> = [];
  let mathOpen = false;
  let mathLines: string[] = [];

  const closeParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  const closeListsTo = (depth = 0) => {
    while (listStack.length > depth) {
      const listType = listStack.pop();
      if (listType) {
        html.push(`</${listType}>`);
      }
    }
  };

  const openListTo = (type: "ul" | "ol", depth: number) => {
    while (listStack.length > depth) {
      const listType = listStack.pop();
      if (listType) {
        html.push(`</${listType}>`);
      }
    }

    if (listStack[depth] && listStack[depth] !== type) {
      closeListsTo(depth);
    }

    while (listStack.length <= depth) {
      html.push(`<${type}>`);
      listStack.push(type);
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (mathOpen) {
      if (trimmed === "$$") {
        html.push(`<div class="math-block">$$${escapeHtml(mathLines.join("\n"))}$$</div>`);
        mathLines = [];
        mathOpen = false;
      } else {
        mathLines.push(line);
      }
      continue;
    }

    if (trimmed === "$$") {
      closeParagraph();
      closeListsTo();
      mathOpen = true;
      continue;
    }

    if (!trimmed) {
      closeParagraph();
      closeListsTo();
      continue;
    }

    if (trimmed.startsWith("#### ")) {
      closeParagraph();
      closeListsTo();
      const heading = trimmed.slice(5);
      html.push(`<h4 id="${slugifyHeading(heading)}">${inlineMarkdown(heading)}</h4>`);
      continue;
    }

    if (trimmed.startsWith("### ")) {
      closeParagraph();
      closeListsTo();
      const heading = trimmed.slice(4);
      html.push(`<h3 id="${slugifyHeading(heading)}">${inlineMarkdown(heading)}</h3>`);
      continue;
    }

    if (trimmed.startsWith("## ")) {
      closeParagraph();
      closeListsTo();
      const heading = trimmed.slice(3);
      html.push(`<h2 id="${slugifyHeading(heading)}">${inlineMarkdown(heading)}</h2>`);
      continue;
    }

    if (trimmed.startsWith("# ")) {
      closeParagraph();
      closeListsTo();
      const heading = trimmed.slice(2);
      html.push(`<h1 id="${slugifyHeading(heading)}">${inlineMarkdown(heading)}</h1>`);
      continue;
    }

    const unorderedListMatch = line.match(/^(\s*)-\s+(.*)$/);
    if (unorderedListMatch) {
      closeParagraph();
      const depth = Math.floor(unorderedListMatch[1].replace(/\t/g, "  ").length / 2);
      openListTo("ul", depth);
      html.push(`<li>${inlineMarkdown(unorderedListMatch[2])}</li>`);
      continue;
    }

    const orderedListMatch = line.match(/^(\s*)\d+\.\s+(.*)$/);
    if (orderedListMatch) {
      closeParagraph();
      const depth = Math.floor(orderedListMatch[1].replace(/\t/g, "  ").length / 2);
      openListTo("ol", depth);
      html.push(`<li>${inlineMarkdown(orderedListMatch[2])}</li>`);
      continue;
    }

    if (trimmed.startsWith(">")) {
      closeParagraph();
      closeListsTo();
      const quote = trimmed.replace(/^>\s?/, "");
      if (quote) {
        html.push(`<blockquote>${inlineMarkdown(quote)}</blockquote>`);
      }
      continue;
    }

    paragraph.push(trimmed);
  }

  closeParagraph();
  closeListsTo();

  return html.join("\n");
}

export function MarkdownView({ content }: MarkdownViewProps) {
  const handleClick = (event: MouseEvent<HTMLDivElement>) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const link = target.closest<HTMLAnchorElement>("a[data-note-anchor]");
    if (!link) {
      return;
    }

    const anchor = link.dataset.noteAnchor?.replace(/^#/, "");
    if (!anchor) {
      return;
    }

    const destination = document.getElementById(anchor);
    if (destination) {
      event.preventDefault();
      destination.scrollIntoView({ behavior: "smooth", block: "start" });
      window.history.replaceState(null, "", window.location.hash || window.location.pathname);
    }
  };

  return (
    <div
      className="markdown-body"
      onClick={handleClick}
      dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
    />
  );
}
