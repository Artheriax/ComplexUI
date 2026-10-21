import { readFile } from "node:fs/promises";

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
  await page.getByPlaceholder("Search templates…").fill(title.source);
  await page.getByRole("button", { name: title }).click();
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

test("uploads, previews, and deletes a workspace dataset", async ({
  page,
  request,
}) => {
  const datasetName = `e2e-iris-${Date.now()}.csv`;

  try {
    await page.getByRole("button", { name: "Datasets" }).click();
    await page.getByLabel("Upload dataset files").setInputFiles({
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

