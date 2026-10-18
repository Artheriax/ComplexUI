/**
 * Local Monaco bundling: avoids the default CDN loader so the editor works
 * offline. Python needs only the base editor worker (tokenization is built in).
 */
import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/editor/editor.worker.js?worker";

self.MonacoEnvironment = {
  getWorker() {
    return new editorWorker();
  },
};

loader.config({ monaco });

export { monaco };
