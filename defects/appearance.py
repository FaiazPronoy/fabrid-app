from defects.palette import MODES, hexcode

MODE_KEY = "fd-mode"
MODE_JS = """
() => {
  const key = 'fd-mode';
  const stored = () => {
    try {
      return localStorage.getItem(key) === 'dark' ? 'dark' : 'light';
    } catch (error) {
      return 'light';
    }
  };
  let mode = stored();
  const apply = () => {
    const dark = mode === 'dark';
    for (const node of [document.documentElement, document.body]) {
      if (node.classList.contains('dark') !== dark) {
        node.classList.toggle('dark', dark);
      }
    }
  };
  window.fdToggleMode = () => {
    mode = mode === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem(key, mode);
    } catch (error) {}
    apply();
  };
  apply();
  const watch = new MutationObserver(apply);
  watch.observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  watch.observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""
TOGGLE_JS = "() => { if (window.fdToggleMode) { window.fdToggleMode(); } }"

THEME_TOKENS = {
    "body_background_fill": "bg",
    "body_text_color": "text",
    "body_text_color_subdued": "muted",
    "background_fill_primary": "card",
    "background_fill_secondary": "bg",
    "block_background_fill": "card",
    "block_border_color": "border",
    "border_color_primary": "border",
    "button_primary_background_fill": "accent-fill",
    "button_primary_background_fill_hover": "accent-hover",
}


def theme_values() -> dict[str, str]:
    values = {"button_primary_text_color": "#ffffff", "button_primary_text_color_dark": "#ffffff"}
    for name, token in THEME_TOKENS.items():
        values[name] = hexcode(MODES["light"][token])
        values[f"{name}_dark"] = hexcode(MODES["dark"][token])
    return values
