import type { WidgetParam } from "../types/graph";

export function isWidgetVisible(
  widget: WidgetParam,
  values: Record<string, unknown>,
  connected: ReadonlySet<string> = new Set(),
): boolean {
  const conditions = widget.visible_when ?? [];
  if (conditions.length === 0) {
    return true;
  }
  if (connected.has(widget.name)) {
    return true;
  }
  if (conditions.some((condition) => connected.has(condition.widget))) {
    return true;
  }
  return conditions.every((condition) =>
    condition.values.some((value) => Object.is(value, values[condition.widget])),
  );
}
