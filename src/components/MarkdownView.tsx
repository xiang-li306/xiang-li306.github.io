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
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2" />')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function renderMarkdown(content: string) {
  const lines = content.split(/\r?\n/);
  const html: string[] = [];
  let paragraph: string[] = [];
  let listOpen = false;
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
      html.push("</ul>");
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
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${inlineMarkdown(trimmed.slice(2))}</li>`);
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
