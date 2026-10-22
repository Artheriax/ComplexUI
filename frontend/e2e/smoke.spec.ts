import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

const executionPanel = (page: import("@playwright/test").Page) =>
  page.getByRole("complementary", { name: "Execution output" });

const canvasStatus = (page: import("@playwright/test").Page) =>
  page.getByTestId("canvas-status");

async function openWorkflow(
  page: import("@playwright/test").Page,
  title: RegExp,
) {
  await page.getByRole("button", { name: "Templates" }).click();
  await page
    .getByPlaceholder("Search templates…")
    .fill(title.source.replace(/\\(.)/g, "$1"));
  await page.getByRole("button", { name: title }).click();
}

async function edgeDragPoint(
  page: import("@playwright/test").Page,
  edgeTestId: string,
) {
  return page
    .locator(`[data-testid="${edgeTestId}"] .react-flow__edge-interaction`)
    .evaluate((element) => {
      const path = element as SVGPathElement;
      const point = path.getPointAtLength(path.getTotalLength() / 2);
      const matrix = path.getScreenCTM();
      if (!matrix) {
        throw new Error("Edge path is not rendered");
      }
      return {
        x: matrix.a * point.x + matrix.c * point.y + matrix.e,
        y: matrix.b * point.x + matrix.d * point.y + matrix.f,
      };
    });
}

async function openFileMenu(page: import("@playwright/test").Page) {
  await page
    .locator(".react-flow__pane")
    .click({ button: "right", position: { x: 160, y: 160 } });
  await page.getByRole("menuitem", { name: "File" }).click();
  await expect(
    page.getByRole("menu", { name: "File actions" }),
  ).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTitle("Backend streaming connected")).toBeVisible();
});

test("shows and hides the system monitor from settings", async ({ page }) => {
  await expect(
    page.getByRole("region", { name: "System monitor" }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).click();
  const settings = page.getByRole("dialog", { name: "Settings" });
  await settings.getByRole("tab", { name: "System" }).click();
  const toggle = settings.getByRole("switch", {
    name: "Show system monitor",
  });
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await toggle.click();
  await settings.getByRole("button", { name: "Close settings" }).click();
  await expect(
    page.getByRole("region", { name: "System monitor" }),
  ).toBeHidden();

  await page.getByRole("button", { name: "Settings" }).click();
  const reopened = page.getByRole("dialog", { name: "Settings" });
  await reopened.getByRole("tab", { name: "System" }).click();
  await reopened
    .getByRole("switch", { name: "Show system monitor" })
    .click();
  await reopened.getByRole("button", { name: "Close settings" }).click();
  await expect(
    page.getByRole("region", { name: "System monitor" }),
  ).toBeVisible();
});

test("lists generated media in the Assets Outputs tab", async ({ page }) => {
  await openWorkflow(page, /Vision Media Smoke Test/);
  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 60_000 },
  );

  await page.getByRole("button", { name: "Assets" }).click();
  await page.getByRole("tab", { name: "Outputs" }).click();
  await expect(
    page.getByRole("button", { name: "Preview vision-demo.png" }),
  ).toBeVisible();
});

test("loads a bundled workflow, updates settings, and runs it", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);
  await expect(
    canvasStatus(page).getByText("4 nodes", { exact: true }),
  ).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).click();
  const settings = page.getByRole("dialog", { name: "Settings" });
  await expect(settings).toBeVisible();

  const showEdges = settings.getByRole("switch", {
    name: "Show connection lines",
  });
  await expect(showEdges).toHaveAttribute("aria-checked", "true");
  await showEdges.click();
  await expect(showEdges).toHaveAttribute("aria-checked", "false");
  await settings.getByRole("button", { name: "Close settings" }).click();

  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 60_000 },
  );
  await expect(executionPanel(page).getByText("accuracy")).toBeVisible();

  await executionPanel(page)
    .getByRole("button", { name: "Run history" })
    .click();
  const history = page.getByRole("dialog", { name: "Run history" });
  await expect(history).toBeVisible();
  await expect(
    history.getByText("Load Iris Dataset", { exact: true }),
  ).toBeVisible();
  await expect(history.getByText(/4\/4 nodes completed/)).toBeVisible();
  await history.getByRole("button", { name: "Close run history" }).click();
  await expect(history).toBeHidden();
});

test("runs an offline image-classification workflow", async ({ page }) => {
  await openWorkflow(page, /Image Classification \(Torchvision\)/);
  await expect(
    canvasStatus(page).getByText("3 nodes", { exact: true }),
  ).toBeVisible();

  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 120_000 },
  );
  await expect(
    executionPanel(page).getByText("5 classifications"),
  ).toBeVisible();

  const demo = page
    .locator(".react-flow__node")
    .filter({ hasText: "Synthetic Demo Image" });
  await demo.click({ force: true, position: { x: 40, y: 10 } });
  const initialBox = await demo.boundingBox();
  expect(initialBox).not.toBeNull();
  await page.mouse.move(initialBox!.x + 40, initialBox!.y + 10);
  await page.mouse.down();
  await page.mouse.move(initialBox!.x - 180, initialBox!.y + 10, {
    steps: 12,
  });
  await page.mouse.up();
  const image = demo.getByAltText("Node output");
  await expect(image).toBeVisible();
  const imageBefore = await image.boundingBox();
  const demoBefore = await demo.boundingBox();
  expect(imageBefore).not.toBeNull();
  const resizeHandle = demo.locator(
    ".react-flow__resize-control.bottom.right",
  );
  const resizeBox = await resizeHandle.boundingBox();
  expect(resizeBox).not.toBeNull();
  await page.mouse.move(
    resizeBox!.x + resizeBox!.width / 2,
    resizeBox!.y + resizeBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    resizeBox!.x + resizeBox!.width / 2 + 160,
    resizeBox!.y + resizeBox!.height / 2 + 100,
    { steps: 12 },
  );
  await page.mouse.up();

  const demoAfter = await demo.boundingBox();
  expect(demoAfter!.width).toBeGreaterThan(demoBefore!.width + 100);
  await expect
    .poll(async () => (await image.boundingBox())?.width ?? 0)
    .toBeGreaterThan(imageBefore!.width + 20);
});

test("resizes a node without scaling parameter controls", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const split = page
    .locator(".react-flow__node")
    .filter({ hasText: "Train / Test Split" });
  await split.click();
  const parameter = split.getByRole("spinbutton").first();
  const parameterBefore = await parameter.evaluate(
    (element) => getComputedStyle(element).width,
  );

  const resizeHandle = split.locator(
    ".react-flow__resize-control.bottom.right",
  );
  const resizeBox = await resizeHandle.boundingBox();
  expect(resizeBox).not.toBeNull();
  const before = await split.boundingBox();
  expect(before).not.toBeNull();
  await page.mouse.move(
    resizeBox!.x + resizeBox!.width / 2,
    resizeBox!.y + resizeBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    resizeBox!.x + resizeBox!.width / 2 + 140,
    resizeBox!.y + resizeBox!.height / 2 + 100,
    { steps: 12 },
  );
  await page.mouse.up();

  await expect
    .poll(async () => (await split.boundingBox())?.width ?? 0)
    .toBeGreaterThan(before!.width + 80);
  const parameterAfter = await parameter.evaluate(
    (element) => getComputedStyle(element).width,
  );
  expect(parameterAfter).toBe(parameterBefore);

  await page.waitForFunction(() =>
    localStorage
      .getItem("complexui.workspace.v1")
      ?.includes('"width"'),
  );
  await page.reload();
  await expect(page.getByTitle("Backend streaming connected")).toBeVisible();
  const restored = page
    .locator(".react-flow__node")
    .filter({ hasText: "Train / Test Split" });
  await expect
    .poll(async () => (await restored.boundingBox())?.width ?? 0)
    .toBeGreaterThan(before!.width + 80);
});

test("runs an offline video-detection workflow", async ({ page }) => {
  await openWorkflow(page, /Video Detection \(YOLO\)/);
  await expect(
    canvasStatus(page).getByText("3 nodes", { exact: true }),
  ).toBeVisible();

  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 120_000 },
  );
  await expect(
    executionPanel(page).getByText(/\d+ detections/),
  ).toBeVisible();
  const video = page
    .locator('video[src="/api/outputs/vision-video-detection.mp4/raw"]')
    .first();
  await expect(video).toBeVisible();
  await expect
    .poll(() =>
      video.evaluate(
        (element) => (element as HTMLVideoElement).readyState,
      ),
    )
    .toBeGreaterThanOrEqual(2);
});

test("does not drag an edge by grabbing its line", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const sourcePoint = await edgeDragPoint(
    page,
    "rf__edge-split:train->knn:train",
  );

  await page.mouse.move(sourcePoint.x, sourcePoint.y);
  await page.mouse.down();
  await page.mouse.move(520, 700, { steps: 16 });
  await page.mouse.up();

  await expect(page.locator("[data-node-search-popup]")).toHaveCount(0);
  await expect(page.locator(".react-flow__edge")).toHaveCount(4);
  await expect
    .poll(() =>
      page
        .locator(".react-flow__edge")
        .evaluateAll((edges) =>
          edges.map((edge) => edge.getAttribute("data-testid")),
        ),
    )
    .toContain("rf__edge-split:train->knn:train");
});

test("drags a connected input to another compatible input", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const knn = page
    .locator(".react-flow__node")
    .filter({ hasText: "KNN Classifier" });
  await knn.click({ position: { x: 40, y: 10 } });
  await page.keyboard.press("Control+d");
  await expect(page.locator(".react-flow__node")).toHaveCount(5);
  await page.getByRole("button", { name: "Fit View" }).click();

  const duplicate = page
    .locator(".react-flow__node.selected")
    .filter({ hasText: "KNN Classifier" });
  await expect(duplicate).toHaveCount(1);
  const duplicateId = await duplicate.getAttribute("data-id");
  expect(duplicateId).not.toBeNull();
  const targetHandle = duplicate.locator(
    '.react-flow__handle-left[data-handleid="train"]',
  );
  await expect(targetHandle).toBeInViewport();
  const targetBox = await targetHandle.boundingBox();
  expect(targetBox).not.toBeNull();
  const sourceHandle = page.locator(
    '.react-flow__node[data-id="knn"] .react-flow__handle-left[data-handleid="train"]',
  );
  await expect(sourceHandle).toBeInViewport();
  const sourceBox = await sourceHandle.boundingBox();
  expect(sourceBox).not.toBeNull();

  await page.mouse.move(
    sourceBox!.x + sourceBox!.width / 2,
    sourceBox!.y + sourceBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    targetBox!.x + targetBox!.width / 2,
    targetBox!.y + targetBox!.height / 2,
    { steps: 16 },
  );
  await expect
    .poll(() => page.locator(".cu-handle-compatible").count())
    .toBeGreaterThan(0);
  const previewLine = page.locator(".cu-edge-retarget-line");
  await expect(previewLine).toBeVisible();
  expect(await previewLine.getAttribute("stroke-dasharray")).toBeNull();
  expect(
    await page.locator(".cu-handle-incompatible").count(),
  ).toBeGreaterThan(0);
  await page.mouse.up();

  const edgeIds = await page
    .locator(".react-flow__edge")
    .evaluateAll((edges) =>
      edges.map((edge) => edge.getAttribute("data-testid")),
    );
  expect(edgeIds).toContain(
    `rf__edge-split:train->${duplicateId}:train`,
  );
  expect(edgeIds).not.toContain("rf__edge-split:train->knn:train");
  await expect(page.locator(".react-flow__edge")).toHaveCount(4);
});

test("drags a connected input to empty canvas and moves it to a new node", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const knn = page
    .locator(".react-flow__node")
    .filter({ hasText: "KNN Classifier" });
  const sourceHandle = knn.locator(
    '.react-flow__handle-left[data-handleid="train"]',
  );
  const sourceBox = await sourceHandle.boundingBox();
  expect(sourceBox).not.toBeNull();

  await page.mouse.move(
    sourceBox!.x + sourceBox!.width / 2,
    sourceBox!.y + sourceBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(520, 700, { steps: 16 });
  await expect
    .poll(() => page.locator(".cu-handle-compatible").count())
    .toBeGreaterThan(0);
  await page.mouse.up();

  const search = page.getByRole("combobox", { name: "Search nodes" });
  await expect(search).toBeVisible();
  await expect(page.locator(".react-flow__edge")).toHaveCount(3);
  await expect
    .poll(() =>
      page
        .locator(".react-flow__edge")
        .evaluateAll((edges) =>
          edges.map((edge) => edge.getAttribute("data-testid")),
        ),
    )
    .not.toContain("rf__edge-split:train->knn:train");
  await search.fill("KNN Classifier");
  await page.getByRole("option", { name: /KNN Classifier/ }).click();

  await expect(page.locator(".react-flow__node")).toHaveCount(5);
  const created = page
    .locator(".react-flow__node.selected")
    .filter({ hasText: "KNN Classifier" });
  await expect(created).toHaveCount(1);
  const createdId = await created.getAttribute("data-id");
  expect(createdId).not.toBeNull();
  await expect(page.locator(".react-flow__edge")).toHaveCount(4);
  await expect
    .poll(() =>
      page
        .locator(".react-flow__edge")
        .evaluateAll((edges) =>
          edges.map((edge) => edge.getAttribute("data-testid")),
        ),
    )
    .toContain(`rf__edge-split:train->${createdId}:train`);
});

test("shift-drags an output bundle to another output", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const split = page
    .locator(".react-flow__node")
    .filter({ hasText: "Train / Test Split" });
  const source = split.locator(
    '.react-flow__handle-right[data-handleid="test"]',
  );
  const target = split.locator(
    '.react-flow__handle-right[data-handleid="train"]',
  );
  const sourceBox = await source.boundingBox();
  const targetBox = await target.boundingBox();
  expect(sourceBox).not.toBeNull();
  expect(targetBox).not.toBeNull();

  await page.keyboard.down("Shift");
  await page.mouse.move(
    sourceBox!.x + sourceBox!.width / 2,
    sourceBox!.y + sourceBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    targetBox!.x + targetBox!.width / 2,
    targetBox!.y + targetBox!.height / 2,
    { steps: 16 },
  );
  await page.mouse.up();
  await page.keyboard.up("Shift");

  await expect
    .poll(() =>
      page
        .locator(".react-flow__edge")
        .evaluateAll((edges) =>
          edges.map((edge) => edge.getAttribute("data-testid")),
        ),
    )
    .toContain("rf__edge-split:train->evaluate:test");
  await expect(page.locator(".react-flow__edge")).toHaveCount(4);
});

test("opens node search when an output bundle is dropped on empty canvas", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const split = page
    .locator(".react-flow__node")
    .filter({ hasText: "Train / Test Split" });
  const source = split.locator(
    '.react-flow__handle-right[data-handleid="test"]',
  );
  const sourceBox = await source.boundingBox();
  expect(sourceBox).not.toBeNull();

  await page.keyboard.down("Shift");
  await page.mouse.move(
    sourceBox!.x + sourceBox!.width / 2,
    sourceBox!.y + sourceBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(520, 700, { steps: 16 });
  await page.mouse.up();
  await page.keyboard.up("Shift");

  await expect(page.locator("[data-node-search-popup]")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator("[data-node-search-popup]")).toHaveCount(0);
  await expect(page.locator(".react-flow__edge")).toHaveCount(4);
  await expect
    .poll(() =>
      page
        .locator(".react-flow__edge")
        .evaluateAll((edges) =>
          edges.map((edge) => edge.getAttribute("data-testid")),
        ),
    )
    .toContain("rf__edge-split:test->evaluate:test");
});

test("aligns selected nodes with the selection toolbar", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);
  await page.keyboard.press("Control+a");

  const toolbar = page.getByRole("toolbar", { name: "Selection tools" });
  await expect(toolbar).toBeVisible();
  await expect(toolbar.getByText("4 selected")).toBeVisible();
  await toolbar.getByRole("button", { name: "Align left" }).click();

  const lefts = await page.locator(".react-flow__node").evaluateAll((elements) =>
    elements.map((element) => Math.round(element.getBoundingClientRect().left)),
  );
  expect(new Set(lefts).size).toBe(1);

  await page.keyboard.press("Control+z");
  const restoredLefts = await page
    .locator(".react-flow__node")
    .evaluateAll((elements) =>
      elements.map((element) =>
        Math.round(element.getBoundingClientRect().left),
      ),
    );
  expect(new Set(restoredLefts).size).toBeGreaterThan(1);
});

test("right-click opens the menu and double-click opens node search", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);
  const pane = page.locator(".react-flow__pane");
  await pane.click({ button: "right", position: { x: 160, y: 160 } });

  const menu = page.getByRole("menu", { name: "Canvas actions" });
  const search = page.locator("[data-node-search-popup]");
  await expect(menu).toBeVisible();
  await expect(search).toHaveCount(0);

  await menu.getByRole("menuitem", { name: /^Select all/ }).click();
  await expect(menu).toHaveCount(0);
  await expect(
    page
      .getByRole("toolbar", { name: "Selection tools" })
      .getByText("4 selected"),
  ).toBeVisible();

  await pane.dblclick({ position: { x: 120, y: 120 } });
  await expect(search).toBeVisible();
  await page.getByRole("button", { name: "Close node search" }).click();
});

test("steps a value node and wires it into a widget parameter", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);

  await page
    .locator(".react-flow__pane")
    .dblclick({ position: { x: 140, y: 140 } });
  const search = page.getByRole("combobox", { name: "Search nodes" });
  await expect(search).toBeVisible();
  await search.fill("Float Value");
  await page.getByRole("option", { name: /Float Value/ }).click();

  const valueNode = page
    .locator(".react-flow__node")
    .filter({ hasText: "Float Value" });
  await expect(valueNode).toBeVisible();
  const valueInput = valueNode.getByRole("spinbutton").first();
  await expect(valueInput).toHaveValue("0");
  await valueInput.fill("0.5");
  await valueNode
    .getByRole("button", { name: "Increase value by 0.1" })
    .click();
  await expect(valueInput).toHaveValue("0.6");
  await valueInput.fill("0.5");
  await page.getByRole("button", { name: "Fit View" }).click();

  const split = page
    .locator(".react-flow__node")
    .filter({ hasText: "Train / Test Split" });
  const sourceHandle = valueNode.locator(
    '.react-flow__handle-right[data-handleid="result"]',
  );
  const targetHandle = split.locator(
    '.react-flow__handle-left[data-handleid="test_size"]',
  );
  await expect(sourceHandle).toBeInViewport();
  await expect(targetHandle).toBeInViewport();
  const sourceBox = await sourceHandle.boundingBox();
  const targetBox = await targetHandle.boundingBox();
  expect(sourceBox).not.toBeNull();
  expect(targetBox).not.toBeNull();
  await page.mouse.move(
    sourceBox!.x + sourceBox!.width / 2,
    sourceBox!.y + sourceBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    targetBox!.x + targetBox!.width / 2,
    targetBox!.y + targetBox!.height / 2,
    { steps: 12 },
  );
  await expect
    .poll(() => page.locator(".cu-handle-compatible").count())
    .toBeGreaterThan(0);
  expect(
    await page.locator(".cu-handle-incompatible").count(),
  ).toBeGreaterThan(0);
  await page.mouse.up();

  await expect(split.getByText("linked", { exact: true })).toBeVisible();
  await expect(page.locator(".react-flow__edge")).toHaveCount(5);

  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 60_000 },
  );
  await expect(split.getByText(/X \[75,4\]/).first()).toBeVisible();
});

test("opens node search when a widget connection is dropped on empty canvas", async ({
  page,
}) => {
  await openWorkflow(page, /Iris KNN Classification/);
  await page.getByRole("button", { name: "Fit View" }).click();

  const split = page
    .locator(".react-flow__node")
    .filter({ hasText: "Train / Test Split" });
  const widgetHandle = split.locator(
    '.react-flow__handle-left[data-handleid="test_size"]',
  );
  await expect(widgetHandle).toBeInViewport();
  const handleBox = await widgetHandle.boundingBox();
  expect(handleBox).not.toBeNull();

  await page.mouse.move(
    handleBox!.x + handleBox!.width / 2,
    handleBox!.y + handleBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(520, 700, { steps: 12 });
  await page.mouse.up();

  const search = page.getByRole("combobox", { name: "Search nodes" });
  await expect(search).toBeVisible();
  await expect(page.getByText("Connect a node")).toBeVisible();
  await search.fill("Float Value");
  await page.getByRole("option", { name: /Float Value/ }).click();

  await expect(split.getByText("linked", { exact: true })).toBeVisible();
  await expect(page.locator(".react-flow__edge")).toHaveCount(5);
});

test("groups and ungroups nodes in a frame", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);
  await page.keyboard.press("Control+a");

  const toolbar = page.getByRole("toolbar", { name: "Selection tools" });
  const frameLabel = page.getByText("Frame 1", { exact: true });
  await toolbar
    .getByRole("button", { name: "Group selection", exact: true })
    .click();

  await expect(frameLabel).toBeVisible();
  await expect(toolbar.getByText("1 selected")).toBeVisible();
  await expect(
    canvasStatus(page).getByText("5 nodes", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".react-flow__node")).toHaveCount(5);

  await page.keyboard.press("Control+z");
  await expect(frameLabel).toHaveCount(0);
  await expect(page.locator(".react-flow__node")).toHaveCount(4);

  await page.keyboard.press("Control+Shift+z");
  await expect(frameLabel).toBeVisible();
  await expect(page.locator(".react-flow__node")).toHaveCount(5);

  await page.waitForFunction(() => {
    const raw = localStorage.getItem("complexui.workflow.v1");
    return (
      raw?.includes('"isFrame":true') && raw.includes('"parentId"')
    );
  });
  await page.reload();
  await expect(page.getByTitle("Backend streaming connected")).toBeVisible();
  await expect(frameLabel).toBeVisible();
  await expect(page.locator(".react-flow__node")).toHaveCount(5);

  await frameLabel.click();
  await expect(toolbar).toBeVisible();
  await toolbar
    .getByRole("button", { name: "Ungroup selection", exact: true })
    .click();
  await expect(frameLabel).toHaveCount(0);
  await expect(page.locator(".react-flow__node")).toHaveCount(4);
});

test("creates, reopens and runs a reusable subgraph", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);
  await page.keyboard.press("Control+a");

  const selection = page.getByRole("toolbar", { name: "Selection tools" });
  await selection
    .getByRole("button", { name: "Create subgraph", exact: true })
    .click();
  await expect(
    canvasStatus(page).getByText("1 node", { exact: true }),
  ).toBeVisible();

  const instance = page
    .locator(".react-flow__node")
    .filter({ hasText: "Subgraph 1" });
  await expect(instance).toBeVisible();
  await instance.dblclick();

  const hierarchy = page.getByRole("navigation", { name: "Canvas hierarchy" });
  await expect(hierarchy).toBeVisible();
  await expect(hierarchy.getByText("Subgraph 1")).toBeVisible();
  await expect(page.getByText("Inputs", { exact: true })).toBeVisible();
  await expect(page.getByText("Outputs", { exact: true })).toBeVisible();
  await expect(canvasStatus(page).getByRole("button", { name: "Run workflow" })).toBeDisabled();

  const inputManager = page
    .locator(".react-flow__node")
    .filter({ hasText: "Inputs" });
  const outputManager = page
    .locator(".react-flow__node")
    .filter({ hasText: "Outputs" });
  const addInputHandle = inputManager
    .locator(".react-flow__handle-right")
    .last();
  const handleBox = await addInputHandle.boundingBox();
  expect(handleBox).not.toBeNull();
  await page.mouse.move(
    handleBox!.x + handleBox!.width / 2,
    handleBox!.y + handleBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(600, 640, { steps: 12 });
  await page.mouse.up();
  await expect(
    page.getByRole("combobox", { name: "Search nodes" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close node search" }).click();

  await page.getByRole("button", { name: "Add input port" }).click();
  await page.getByRole("button", { name: "Add output port" }).click();
  await expect(inputManager.getByText("input", { exact: true })).toBeVisible();
  await expect(
    outputManager.getByText("output", { exact: true }),
  ).toBeVisible();

  await hierarchy.getByRole("button", { name: "Root", exact: true }).click();
  await page.waitForFunction(() =>
    localStorage
      .getItem("complexui.workflow.v1")
      ?.includes('"subgraphs"'),
  );
  await page.reload();
  await expect(page.getByTitle("Backend streaming connected")).toBeVisible();
  const restored = page
    .locator(".react-flow__node")
    .filter({ hasText: "Subgraph 1" });
  await expect(restored).toBeVisible();
  await restored.dblclick();
  await expect(page.getByText("Inputs", { exact: true })).toBeVisible();
  await expect(
    page
      .locator(".react-flow__node")
      .filter({ hasText: "Inputs" })
      .getByText("input", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".react-flow__node")
      .filter({ hasText: "Outputs" })
      .getByText("output", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "Canvas hierarchy" })
    .getByRole("button", { name: "Root", exact: true })
    .click();

  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 60_000 },
  );
  await expect(executionPanel(page).getByText("accuracy")).toBeVisible();

  await restored.click();
  await page.keyboard.press("Delete");
  await expect(restored).toHaveCount(0);
  await page.getByRole("button", { name: "Nodes" }).click();
  await expect(page.getByText("Subgraphs", { exact: true })).toHaveCount(0);
});

test("exports Python and notebook artifacts", async ({ page }) => {
  await openWorkflow(page, /Iris KNN Classification/);

  await openFileMenu(page);
  const pythonDownloadPromise = page.waitForEvent("download");
  await page.getByRole("menuitem", { name: "Export Python script" }).click();
  const pythonDownload = await pythonDownloadPromise;
  expect(pythonDownload.suggestedFilename()).toMatch(/\.py$/);
  const pythonPath = await pythonDownload.path();
  if (!pythonPath) {
    throw new Error("Python export did not produce a local file");
  }
  const pythonSource = await readFile(pythonPath, "utf8");
  expect(pythonSource).toContain("# /// script");
  expect(pythonSource).toContain("def __node_exec_load");

  await openFileMenu(page);
  const notebookDownloadPromise = page.waitForEvent("download");
  await page
    .getByRole("menuitem", { name: "Export Jupyter notebook" })
    .click();
  const notebookDownload = await notebookDownloadPromise;
  expect(notebookDownload.suggestedFilename()).toMatch(/\.ipynb$/);
  const notebookPath = await notebookDownload.path();
  if (!notebookPath) {
    throw new Error("Notebook export did not produce a local file");
  }
  const notebook = JSON.parse(await readFile(notebookPath, "utf8")) as {
    nbformat: number;
    cells: unknown[];
  };
  expect(notebook.nbformat).toBe(4);
  expect(notebook.cells.length).toBeGreaterThan(2);
});

test("saves, reloads and reopens a named project", async ({
  page,
  request,
}) => {
  const projectName = `E2E Project ${Date.now()}`;

  try {
    await openWorkflow(page, /Iris KNN Classification/);
    await expect(
      canvasStatus(page).getByText("4 nodes", { exact: true }),
    ).toBeVisible();

    await page.keyboard.press("Control+s");
    const dialog = page.getByRole("dialog", { name: "Name this project" });
    await expect(dialog).toBeVisible();
    await dialog.getByLabel("Project name").fill(projectName);
    await dialog.getByRole("button", { name: "Save project" }).click();
    await expect(dialog).toBeHidden();
    await expect(
      page.getByRole("tab", { name: new RegExp(projectName) }),
    ).toBeVisible();

    await page.reload();
    await expect(page.getByTitle("Backend streaming connected")).toBeVisible();
    await expect(
      page.getByRole("tab", { name: new RegExp(projectName) }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Workflows" }).click();
    await page
      .getByRole("button", { name: new RegExp(`^${projectName}`) })
      .click();
    await expect(
      canvasStatus(page).getByText("4 nodes", { exact: true }),
    ).toBeVisible();

    const saved = page.waitForResponse(
      (response) =>
        response.url().includes("/api/projects/") &&
        response.request().method() === "PUT",
    );
    await page.keyboard.press("Control+s");
    expect((await saved).ok()).toBe(true);

    await expect(page.getByText("1 version", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "History", exact: true }).click();
    const history = page.getByRole("dialog", { name: /Version history/ });
    await expect(history).toBeVisible();
    await history.getByRole("button", { name: "Open" }).click();
    await expect(history).toBeHidden();
    await expect(
      canvasStatus(page).getByText("4 nodes", { exact: true }),
    ).toBeVisible();
  } finally {
    const listed = await request.get("/api/projects");
    if (listed.ok()) {
      const projects = (await listed.json()) as { slug: string; name: string }[];
      const project = projects.find((candidate) => candidate.name === projectName);
      if (project) {
        await request.delete(`/api/projects/${project.slug}`);
      }
    }
    await request.put("/api/session", {
      data: {
        workflow: {
          version: 1,
          savedAt: "",
          nodes: [],
          edges: [],
        },
        project_slug: null,
        project_name: null,
      },
    });
  }
});

test("auto-refreshes the model library and selects a local fallback", async ({
  page,
}) => {
  const modelsDir = path.resolve(
    process.cwd(),
    "..",
    "models",
    "classification",
  );
  const modelName = `e2e-model-${Date.now()}.onnx`;
  const modelPath = path.join(modelsDir, modelName);
  await mkdir(modelsDir, { recursive: true });

  try {
    await page.getByRole("button", { name: "Assets" }).click();
    await page.getByRole("tab", { name: "Models" }).click();
    await expect(page.getByLabel("Search models")).toBeVisible();

    await writeFile(modelPath, "e2e model placeholder");
    await expect(page.getByText(modelName, { exact: true })).toBeVisible({
      timeout: 10_000,
    });

    await page.keyboard.press("r");
    await expect(page.getByText(modelName, { exact: true })).toBeVisible();

    await openWorkflow(page, /Image Classification \(Torchvision\)/);
    const field = page.getByLabel("Choose model for local_model");
    await expect(field).toContainText("Built-in fallback");
    await field.click();

    const picker = page.getByRole("dialog", { name: "Choose local model" });
    await expect(picker.getByText(modelName, { exact: true })).toBeVisible();
    await picker.getByText(modelName, { exact: true }).click();
    await expect(field).toContainText(modelName);

    await field.click();
    await page
      .getByRole("dialog", { name: "Choose local model" })
      .getByText("Built-in fallback", { exact: true })
      .click();
    await expect(field).toContainText("Built-in fallback");
  } finally {
    await unlink(modelPath).catch(() => undefined);
  }
});

test("uploads, previews, and deletes a workspace dataset", async ({
  page,
  request,
}) => {
  const datasetName = `e2e-iris-${Date.now()}.csv`;

  try {
    await page.getByRole("button", { name: "Assets" }).click();
    await page.getByLabel("Upload asset files").setInputFiles({
      name: datasetName,
      mimeType: "text/csv",
      buffer: Buffer.from("sepal_length,species\n5.1,setosa\n6.3,virginica\n"),
    });

    await expect(
      page.getByText(datasetName, { exact: true }),
    ).toBeVisible();
    await page.getByLabel(`Preview ${datasetName}`).click();
    const preview = page.getByRole("dialog", {
      name: new RegExp(`Dataset preview.*${datasetName.replace(".", "\\.")}`),
    });
    await expect(preview).toBeVisible();
    await expect(preview.getByText("sepal_length")).toBeVisible();
    await expect(preview.getByText("5.1", { exact: true })).toBeVisible();
    await preview.getByRole("button", { name: "Close dataset preview" }).click();
    await expect(preview).toBeHidden();

    page.once("dialog", (dialog) => void dialog.accept());
    await page.getByLabel(`Delete ${datasetName}`).click();
    await expect(page.getByText(datasetName, { exact: true })).toHaveCount(0);
  } finally {
    await request.delete(`/api/uploads/${encodeURIComponent(datasetName)}`);
  }
});

test("approves modified scratch code and executes it", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.getByRole("button", { name: "Nodes" }).click();
  await page.getByRole("button", { name: /New scratch node/ }).click();
  await expect(
    canvasStatus(page).getByText("1 node", { exact: true }),
  ).toBeVisible();

  await page
    .getByRole("button", { name: "Edit code for Scratch node" })
    .click();
  await page.getByRole("button", { name: /Review & approve/ }).click();
  const review = page.getByRole("dialog", {
    name: /Security review/,
  });
  await expect(review).toBeVisible();
  await review
    .getByRole("button", { name: "Approve & use modified code" })
    .click();
  await expect(review).toBeHidden();

  await canvasStatus(page).getByRole("button", { name: "Run workflow" }).click();
  await expect(executionPanel(page).getByRole("status")).toHaveText(
    "completed",
    { timeout: 30_000 },
  );
  await expect(executionPanel(page).getByText("result")).toBeVisible();
  await expect(executionPanel(page).getByText("42", { exact: true })).toBeVisible();
  expect(pageErrors).toEqual([]);
});

