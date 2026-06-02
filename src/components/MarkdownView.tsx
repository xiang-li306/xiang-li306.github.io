type MarkdownViewProps = {
  content: string;
};

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
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function renderMarkdown(content: string) {
  const lines = content.split(/\r?\n/);
  const html: string[] = [];
  let paragraph: string[] = [];
  let listOpen: "ul" | "ol" | false = false;
  let mathOpen = false;
  let mathLines: string[] = [];

  const closeParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  const closeList = () => {
    if (listOpen) {
      html.push(`</${listOpen}>`);
      listOpen = false;
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
      closeList();
      mathOpen = true;
      continue;
    }

    if (!trimmed) {
      closeParagraph();
      closeList();
      continue;
    }

    if (trimmed.startsWith("#### ")) {
      closeParagraph();
      closeList();
      html.push(`<h4>${inlineMarkdown(trimmed.slice(5))}</h4>`);
      continue;
    }

    if (trimmed.startsWith("### ")) {
      closeParagraph();
      closeList();
      html.push(`<h3>${inlineMarkdown(trimmed.slice(4))}</h3>`);
      continue;
    }

    if (trimmed.startsWith("## ")) {
      closeParagraph();
      closeList();
      html.push(`<h2>${inlineMarkdown(trimmed.slice(3))}</h2>`);
      continue;
    }

    if (trimmed.startsWith("# ")) {
      closeParagraph();
      closeList();
      html.push(`<h1>${inlineMarkdown(trimmed.slice(2))}</h1>`);
      continue;
    }

    if (trimmed.startsWith("- ")) {
      closeParagraph();
      if (listOpen && listOpen !== "ul") {
        closeList();
      }
      if (!listOpen) {
        html.push("<ul>");
        listOpen = "ul";
      }
      html.push(`<li>${inlineMarkdown(trimmed.slice(2))}</li>`);
      continue;
    }

    const orderedListMatch = trimmed.match(/^\d+\.\s+(.*)$/);
    if (orderedListMatch) {
      closeParagraph();
      if (listOpen && listOpen !== "ol") {
        closeList();
      }
      if (!listOpen) {
        html.push("<ol>");
        listOpen = "ol";
      }
      html.push(`<li>${inlineMarkdown(orderedListMatch[1])}</li>`);
      continue;
    }

    if (trimmed.startsWith(">")) {
      closeParagraph();
      closeList();
      const quote = trimmed.replace(/^>\s?/, "");
      if (quote) {
        html.push(`<blockquote>${inlineMarkdown(quote)}</blockquote>`);
      }
      continue;
    }

    paragraph.push(trimmed);
  }

  closeParagraph();
  closeList();

  return html.join("\n");
}

export function MarkdownView({ content }: MarkdownViewProps) {
  return (
    <div
      className="markdown-body"
      dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
    />
  );
}
