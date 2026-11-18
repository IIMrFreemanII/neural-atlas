// The app is client rendered: every visual needs WebGPU, which only exists in the browser.
export const ssr = false;
// Every page is still written out as its own HTML file at build time (an empty shell that the app fills in
// the browser), so a static host answers a link straight to any page with 200 and not with its 404 page.
export const prerender = true;
