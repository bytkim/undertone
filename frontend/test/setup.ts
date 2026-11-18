import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  localStorage.clear();   // the page's settings are kept there: no test inherits another's
  window.history.replaceState(null, "", window.location.pathname);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
Element.prototype.scrollTo = function () {};
