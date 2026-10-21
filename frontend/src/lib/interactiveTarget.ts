const NODE_EDITOR_BLOCKERS = [
  "button",
  "input",
  "textarea",
  "select",
  "a",
  "label",
  "[role='button']",
  "[role='tab']",
  "[contenteditable='true']",
  "[data-node-editor-ignore]",
].join(", ");

/**
 * Return true when a pointer event originated from a control that should keep
 * its native interaction instead of triggering the node code editor.
 */
export function isInteractiveControlTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest(NODE_EDITOR_BLOCKERS) !== null
  );
}
