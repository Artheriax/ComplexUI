import type { TemplateInfo } from "../types/graph";

export const CATEGORY_ORDER = [
  "Values",
  "Data Loaders",
  "Data Cleaning",
  "EDA",
  "Preprocessing",
  "Imbalanced Data",
  "Models",
  "Evaluation",
  "PyTorch",
  "Computer Vision",
];

export interface TemplateGroup {
  category: string;
  items: TemplateInfo[];
}

function searchableText(template: TemplateInfo): string {
  return `${template.title} ${template.name} ${template.description} ${template.category}`.toLowerCase();
}

export function filterTemplates(
  templates: TemplateInfo[],
  query: string,
): TemplateInfo[] {
  const needle = query.trim().toLowerCase();
  return templates.filter(
    (template) =>
      !template.hidden &&
      (needle.length === 0 || searchableText(template).includes(needle)),
  );
}

export function orderTemplates(templates: TemplateInfo[]): TemplateInfo[] {
  const order = new Map(
    CATEGORY_ORDER.map((category, index) => [category, index]),
  );
  return [...templates].sort((left, right) => {
    const leftIndex = order.get(left.category) ?? CATEGORY_ORDER.length;
    const rightIndex = order.get(right.category) ?? CATEGORY_ORDER.length;
    if (leftIndex !== rightIndex) {
      return leftIndex - rightIndex;
    }
    if (left.category !== right.category) {
      return left.category.localeCompare(right.category);
    }
    return left.title.localeCompare(right.title);
  });
}

export function searchTemplates(
  templates: TemplateInfo[],
  query: string,
): TemplateInfo[] {
  return orderTemplates(filterTemplates(templates, query));
}

export function groupTemplates(
  templates: TemplateInfo[],
  query: string,
): TemplateGroup[] {
  const buckets = new Map<string, TemplateInfo[]>();
  for (const template of searchTemplates(templates, query)) {
    const items = buckets.get(template.category);
    if (items) {
      items.push(template);
    } else {
      buckets.set(template.category, [template]);
    }
  }
  const known = CATEGORY_ORDER.filter((category) => buckets.has(category));
  const unknown = [...buckets.keys()]
    .filter((category) => !CATEGORY_ORDER.includes(category))
    .sort();
  return [...known, ...unknown].map((category) => ({
    category,
    items: buckets.get(category) ?? [],
  }));
}
