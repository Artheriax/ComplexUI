export const NODE_SEARCH_WIDTH = 320;
export const NODE_SEARCH_HEIGHT = 400;
export const NODE_SEARCH_MARGIN = 8;

export function nodeSearchPlacement(anchor: { x: number; y: number }): {
  left: number;
  top: number;
} {
  return {
    left: Math.max(
      NODE_SEARCH_MARGIN,
      Math.min(anchor.x, window.innerWidth - NODE_SEARCH_WIDTH - 16),
    ),
    top: Math.max(
      NODE_SEARCH_MARGIN,
      Math.min(anchor.y, window.innerHeight - NODE_SEARCH_HEIGHT),
    ),
  };
}
