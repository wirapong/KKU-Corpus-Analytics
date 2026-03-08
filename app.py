from __future__ import annotations

import html
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from antconc_optimized import (
    AnalyzerConfig,
    CorpusAnalysisError,
    CorpusAnalyzer,
    DEFAULT_STOPWORDS,
    MULTILINGUAL_STOPWORDS,
    REGEX_PRESETS,
    REGEX_REPLACE_PRESETS,
    THAI_STOPWORDS,
)

st.set_page_config(page_title="KKU Corpus Analytics GUI", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 0.9rem; padding-bottom: 1rem;}
    .app-title {font-size: 2rem; font-weight: 800; margin-bottom: .15rem;}
    .app-subtitle {color: #666; margin-bottom: .7rem;}
    .kwic-row {
        display:grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items:center;
        font-family: Consolas, 'SFMono-Regular', Menlo, monospace;
        border:1px solid rgba(120,120,120,.18); border-radius:10px; padding:8px 10px; margin-bottom:6px;
        background: rgba(248,248,248,.8);
    }
    .kwic-left {text-align:right; color:#555; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;}
    .kwic-right {text-align:left; color:#111; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;}
    .kwic-key {padding:2px 8px; border-radius:6px; font-weight:700; color:#111;}
    .mini-note {color:#777; font-size:.92rem; margin:.15rem 0 .4rem 0;}
    .fileview-box {font-family: Consolas, 'SFMono-Regular', Menlo, monospace; white-space: pre-wrap; border:1px solid rgba(120,120,120,.18); border-radius:10px; padding:12px;}
    .toolbar-note {color:#666; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

HIGHLIGHT_COLORS = ["#ffe08a", "#a8e6cf", "#bde0fe", "#f1c0e8", "#ffcad4", "#caffbf", "#ffd6a5", "#d0f4de"]


def uploaded_files_to_analyzer(uploaded_files, *, lowercase: bool, language: str, thai_engine: str) -> CorpusAnalyzer:
    config = AnalyzerConfig(lowercase=lowercase, language=language, thai_engine=thai_engine)
    analyzer = CorpusAnalyzer(config=config)
    if not uploaded_files:
        raise CorpusAnalysisError("Please upload at least one .txt file.")
    loaded = 0
    errors = []
    for f in uploaded_files:
        try:
            text = f.getvalue().decode("utf-8", errors="ignore")
            analyzer.add_text(text=text, name=f.name, path=f.name)
            loaded += 1
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
    if loaded == 0:
        raise CorpusAnalysisError("No files could be loaded.\n" + "\n".join(errors))
    return analyzer


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def sanitize_sheet_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_ก-๙]+", "_", str(name))[:31] or "sheet"


def results_to_excel_bytes(results: Dict[str, pd.DataFrame], metadata: Optional[pd.DataFrame] = None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if metadata is not None and not metadata.empty:
            metadata.to_excel(writer, sheet_name="metadata_summary", index=False)
        used = set()
        for sheet_name, df in results.items():
            safe = sanitize_sheet_name(sheet_name)
            if safe in used:
                i = 2
                while f"{safe[:28]}_{i}" in used:
                    i += 1
                safe = f"{safe[:28]}_{i}"
            used.add(safe)
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(writer, sheet_name=safe, index=False)
    output.seek(0)
    return output.getvalue()


def make_project_zip() -> bytes:
    output = io.BytesIO()
    base = Path(__file__).resolve().parent
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ["app.py", "antconc_optimized.py", "requirements.txt", "README.md"]:
            p = base / name
            if p.exists():
                zf.write(p, arcname=name)
    output.seek(0)
    return output.getvalue()


def pick_stopwords(mode: str, use_stopwords: bool) -> set[str]:
    if not use_stopwords:
        return set()
    mode = (mode or "auto").lower()
    if mode == "thai":
        return THAI_STOPWORDS
    if mode in {"multilingual", "auto"}:
        return MULTILINGUAL_STOPWORDS
    return DEFAULT_STOPWORDS


def parse_patterns(raw_query: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;\n]+", str(raw_query or "")) if p.strip()]


def pattern_color_map(patterns: list[str]) -> dict[str, str]:
    return {pat: HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)] for i, pat in enumerate(patterns)}


def apply_table_controls(df: pd.DataFrame, key: str, sort_default: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        st.info("No rows to display.")
        return pd.DataFrame(columns=df.columns if isinstance(df, pd.DataFrame) else [])
    out = df.copy()
    with st.expander("Sort / Filter / Preview", expanded=False):
        c1, c2, c3 = st.columns([1.1, 1.1, 1.4])
        sort_col = c1.selectbox(
            "Sort by",
            options=["(none)"] + list(out.columns),
            index=(list(out.columns).index(sort_default) + 1 if sort_default in out.columns else 0),
            key=f"sort_{key}",
        )
        ascending = c2.selectbox("Order", ["Descending", "Ascending"], index=0, key=f"asc_{key}") == "Ascending"
        contains = c3.text_input("Contains filter", key=f"filter_{key}", placeholder="type a keyword to filter visible rows")
        if contains:
            mask = out.astype(str).apply(lambda s: s.str.contains(contains, case=False, na=False, regex=False))
            out = out[mask.any(axis=1)]
        if sort_col != "(none)" and sort_col in out.columns:
            try:
                out = out.sort_values(sort_col, ascending=ascending, kind="stable")
            except Exception:
                pass
        out = out.reset_index(drop=True)
        row_count = len(out)
        if row_count == 0:
            st.warning("No rows match the current filter.")
            return out
        if row_count <= 10:
            max_rows = row_count
        else:
            preview_cap = max(10, min(5000, row_count))
            default_rows = min(200, preview_cap)
            max_rows = st.slider("Preview rows", 10, preview_cap, default_rows, key=f"rows_{key}")
        st.caption(f"Showing {min(max_rows, row_count):,} of {row_count:,} rows")
        out = out.head(max_rows)
    return out


def highlight_terms(text: str, color_map: dict[str, str]) -> str:
    safe = html.escape(str(text))
    if not safe or not color_map:
        return safe
    # longer terms first
    for term, color in sorted(color_map.items(), key=lambda x: len(x[0]), reverse=True):
        escaped_term = html.escape(str(term))
        if not escaped_term:
            continue
        safe = re.sub(
            re.escape(escaped_term),
            lambda m, c=color: f"<mark style='background:{c}; padding:0 .15rem; border-radius:.2rem'>{m.group(0)}</mark>",
            safe,
            flags=re.IGNORECASE,
        )
    return safe


def render_kwic_centered(df: pd.DataFrame, color_map: dict[str, str]) -> None:
    if df.empty:
        st.info("No KWIC rows found.")
        return
    st.caption("Centered KWIC view")
    for _, row in df.iterrows():
        left = html.escape(str(row.get("left_context", ""))[-140:])
        keyword = html.escape(str(row.get("keyword", "")))
        matched_pattern = str(row.get("matched_pattern", keyword))
        color = color_map.get(matched_pattern, "#ffe08a")
        right = html.escape(str(row.get("right_context", ""))[:140])
        meta = html.escape(f"{row.get('document', '')} | token #{row.get('token_index', '')} | pattern: {matched_pattern}")
        st.markdown(
            f"<div class='kwic-row'><div class='kwic-left'>{left}</div><div class='kwic-key' style='background:{color}'>{keyword}</div><div class='kwic-right'>{right}</div></div><div class='mini-note'>{meta}</div>",
            unsafe_allow_html=True,
        )


def render_kwic_selector(df: pd.DataFrame, key: str = "kwic_selector") -> Optional[pd.Series]:
    if df.empty:
        return None
    show_cols = [c for c in ["document", "matched_pattern", "token_index", "keyword", "left_context", "right_context"] if c in df.columns]
    show_df = df[show_cols].copy()
    show_df.insert(0, "open", False)
    edited = st.data_editor(
        show_df,
        hide_index=True,
        use_container_width=True,
        disabled=[c for c in show_df.columns if c != "open"],
        key=key,
        column_config={"open": st.column_config.CheckboxColumn("Open in File View")},
    )
    selected = edited.index[edited["open"]].tolist() if "open" in edited.columns else []
    if not selected:
        return None
    return df.iloc[int(selected[0])]


def render_file_view_row(row: pd.Series, color_map: dict[str, str]) -> None:
    matched_pattern = str(row.get("matched_pattern", row.get("keyword", "")))
    snippet = highlight_terms(str(row.get("snippet", "")), {str(row.get("keyword", "")): color_map.get(matched_pattern, "#ffe08a")})
    st.markdown(f"<div class='fileview-box'>{snippet}</div>", unsafe_allow_html=True)
    st.caption(f"{row.get('document', '')} | chars {row.get('char_start', '')}–{row.get('char_end', '')} | pattern: {matched_pattern}")


def render_distribution_plot(df: pd.DataFrame, key: str = "plot") -> Optional[pd.Series]:
    if df is None or df.empty:
        st.info("No distribution data to plot.")
        return None
    plot_df = df.copy()
    fig = px.scatter(
        plot_df,
        x="position_pct",
        y="document",
        color="matched_pattern" if "matched_pattern" in plot_df.columns else "keyword",
        hover_data=[c for c in ["document", "keyword", "matched_pattern", "token_index", "position_pct"] if c in plot_df.columns],
        labels={"position_pct": "Position in document (%)", "document": "Document"},
    )
    fig.update_traces(marker={"symbol": "line-ns-open", "size": 18, "line": {"width": 2}})
    fig.update_layout(height=max(280, 75 + 40 * plot_df["document"].nunique()), xaxis_range=[0, 100], legend_title_text="Pattern")
    selected_row = None
    try:
        event = st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun")
        points = (event or {}).get("selection", {}).get("points", []) if isinstance(event, dict) else []
        if points:
            idx = int(points[0].get("point_index", 0))
            if 0 <= idx < len(plot_df):
                selected_row = plot_df.iloc[idx]
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_fallback")
    return selected_row


def build_metadata_summary(analyzer: CorpusAnalyzer, reference_analyzer: Optional[CorpusAnalyzer], *, query: str, search_mode: str, language: str, thai_engine: str, lowercase: bool, use_stopwords: bool, window: int, context_chars: int, max_rows: int) -> pd.DataFrame:
    token_table = analyzer._token_table_frame()
    rows = [
        {"field": "documents", "value": len(analyzer.documents)},
        {"field": "tokens", "value": len(token_table)},
        {"field": "unique_tokens", "value": int(token_table["token"].nunique() if not token_table.empty else 0)},
        {"field": "query", "value": query},
        {"field": "search_mode", "value": search_mode},
        {"field": "language", "value": language},
        {"field": "thai_engine", "value": thai_engine},
        {"field": "lowercase", "value": lowercase},
        {"field": "use_stopwords", "value": use_stopwords},
        {"field": "kwic_window", "value": window},
        {"field": "context_chars", "value": context_chars},
        {"field": "max_rows", "value": max_rows},
        {"field": "reference_documents", "value": 0 if reference_analyzer is None else len(reference_analyzer.documents)},
    ]
    return pd.DataFrame(rows)


def build_all_results(analyzer: CorpusAnalyzer, reference_analyzer: Optional[CorpusAnalyzer], *, query: str, regex: bool, wildcard: bool, stopwords: set[str], window: int, context_chars: int, max_rows: int) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {
        "basic_stats": analyzer.basic_stats(),
        "word_list": analyzer.word_list(min_freq=1, stopwords=stopwords),
        "ngrams_2": analyzer.ngrams(n=2, min_freq=2, stopwords=stopwords),
    }
    if query.strip():
        kwic = analyzer.concordance(query=query.strip(), window=window, regex=regex, wildcard=wildcard, max_rows=max_rows)
        results["concordance"] = kwic
        results["concordance_plot"] = analyzer.concordance_plot(query=query.strip(), regex=regex, wildcard=wildcard)
        results["file_view"] = analyzer.file_view(query=query.strip(), regex=regex, wildcard=wildcard, context_chars=context_chars, max_rows=max_rows)
        results["collocates"] = analyzer.collocates(query=query.strip(), window=window, min_freq=2, stopwords=stopwords, regex=regex, wildcard=wildcard)
        for side in ["L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5"]:
            results[f"kwic_{side.lower()}"] = analyzer.kwic_sort(kwic, sort_by=side)
    if reference_analyzer is not None:
        results["keyword_list"] = analyzer.keyword_list(reference_analyzer, min_freq=2)
    return results

st.markdown("<div class='app-title'></div>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>KKU Corpus Analyzer</div>", unsafe_allow_html=True)
st.write(
    "Analyze uploaded text corpora with a clean GUI inspired by AntConc. "   
)
st.markdown("<div class='app-subtitle'>Desktop-like sidebar, centered KWIC, clickable distribution plot, multi-color highlighting, Thai-aware tokenization, and workbook export with metadata summary.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Corpus")
    uploaded_files = st.file_uploader("Upload target corpus (.txt)", type=["txt"], accept_multiple_files=True)
    ref_files = st.file_uploader("Upload reference corpus (.txt) for Keyword List", type=["txt"], accept_multiple_files=True)

    st.markdown("### Tokenization")
    language = st.selectbox("Language mode", ["auto", "english", "thai", "multilingual"], index=0)
    thai_engine = st.selectbox("Thai tokenizer", ["newmm", "mm", "longest"], index=0)
    lowercase = st.checkbox("Lowercase", value=True)
    use_stopwords = st.checkbox("Use stopword list", value=True)

    st.markdown("### Search")
    query = st.text_area("Search term(s)", placeholder="e.g. learn\nstudy*\nศึกษา*", height=90)
    st.caption("Multi-search: separate terms with comma, semicolon, or new line.")
    search_mode = st.radio("Search mode", ["exact", "wildcard", "regex"], index=0)
    window = st.slider("KWIC window", 1, 15, 5)
    max_rows = st.slider("Max search rows", 50, 5000, 500)
    context_chars = st.slider("File View context chars", 40, 600, 120)

    st.markdown("### Regex presets")
    preset = st.selectbox("Pattern preset", ["(none)"] + list(REGEX_PRESETS.keys()), index=0)
    if preset != "(none)":
        st.code(REGEX_PRESETS[preset].replace("TERM", (query.splitlines()[0].strip() if query.strip() else "term")), language="regex")
    replace_preset = st.selectbox("Regex replace preset", ["(none)"] + list(REGEX_REPLACE_PRESETS.keys()), index=0)
    if replace_preset != "(none)":
        src = query or "term"
        pat, repl = REGEX_REPLACE_PRESETS[replace_preset]
        try:
            transformed = re.sub(pat, repl, src)
        except re.error:
            transformed = src
        st.code(transformed, language="regex")
        if st.button("Use transformed pattern"):
            st.session_state["applied_regex_pattern"] = transformed

    st.markdown("### Tools")
    tool = st.radio("Select tool", ["Dashboard", "Word List", "Concordance", "Concordance Plot", "File View", "Clusters / N-Grams", "Collocates", "Keyword List"], index=0)
    st.download_button("Download app source (zip)", make_project_zip(), file_name="antconc_streamlit_app.zip", mime="application/zip")

results_store: Dict[str, pd.DataFrame] = {}
try:
    analyzer = uploaded_files_to_analyzer(uploaded_files, lowercase=lowercase, language=language, thai_engine=thai_engine) if uploaded_files else None
    reference_analyzer = uploaded_files_to_analyzer(ref_files, lowercase=lowercase, language=language, thai_engine=thai_engine) if ref_files else None
except Exception as exc:
    analyzer = None
    reference_analyzer = None
    st.error(str(exc))

if analyzer is None:
    st.info("Upload one or more .txt files to begin.")
    st.stop()

if st.session_state.get("applied_regex_pattern") and search_mode == "regex":
    query = st.session_state["applied_regex_pattern"]

stopwords = pick_stopwords(language, use_stopwords)
regex = search_mode == "regex"
wildcard = search_mode == "wildcard"
patterns = parse_patterns(query)
color_map = pattern_color_map(patterns)
metadata_df = build_metadata_summary(
    analyzer,
    reference_analyzer,
    query=query,
    search_mode=search_mode,
    language=language,
    thai_engine=thai_engine,
    lowercase=lowercase,
    use_stopwords=use_stopwords,
    window=window,
    context_chars=context_chars,
    max_rows=max_rows,
)

clegend = st.container()
with clegend:
    if color_map:
        st.markdown("**Pattern colors**")
        chips = " ".join([f"<span style='background:{c};padding:.2rem .45rem;border-radius:.45rem;margin-right:.35rem;display:inline-block'>{html.escape(p)}</span>" for p, c in color_map.items()])
        st.markdown(chips, unsafe_allow_html=True)

if tool == "Dashboard":
    stats_df = analyzer.basic_stats()
    token_table = analyzer._token_table_frame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(analyzer.documents))
    c2.metric("Tokens", len(token_table))
    c3.metric("Unique tokens", token_table["token"].nunique() if not token_table.empty else 0)
    c4.metric("Language mode", language)
    results_store["basic_stats"] = stats_df
    preview = apply_table_controls(stats_df, "basic_stats", sort_default="total_words")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    word_df = analyzer.word_list(min_freq=1, stopwords=stopwords)
    results_store["word_list"] = word_df
    st.markdown("#### Top Word List")
    st.dataframe(apply_table_controls(word_df.head(1000), "top_words", sort_default="frequency"), use_container_width=True, hide_index=True)

elif tool == "Word List":
    min_freq = st.number_input("Minimum frequency", 1, 100000, 1)
    keep_numbers = st.checkbox("Keep numeric tokens", value=True)
    df = analyzer.word_list(min_freq=int(min_freq), stopwords=stopwords, keep_numbers=keep_numbers)
    results_store["word_list"] = df
    st.dataframe(apply_table_controls(df, "word_list", sort_default="frequency"), use_container_width=True, hide_index=True)
    st.download_button("Download CSV", to_csv_bytes(df), file_name="word_list.csv", mime="text/csv")

elif tool == "Concordance":
    if not query.strip():
        st.warning("Enter a search term first.")
    else:
        sort_kwic = st.selectbox("KWIC sort", ["position", "L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5"], index=0)
        df = analyzer.concordance(query=query.strip(), window=window, regex=regex, wildcard=wildcard, max_rows=max_rows)
        df = analyzer.kwic_sort(df, sort_by=sort_kwic)
        results_store["concordance"] = df
        preview = apply_table_controls(df, "concordance", sort_default="token_index")
        tab1, tab2 = st.tabs(["Centered KWIC", "Interactive Table"])
        with tab1:
            render_kwic_centered(preview, color_map)
        with tab2:
            selected = render_kwic_selector(preview, key="kwic_selector_table")
            if selected is not None:
                st.markdown("#### File View at selected KWIC position")
                fv = analyzer.file_view_at(int(selected["doc_id"]), int(selected["char_start"]), int(selected["char_end"]), context_chars=context_chars)
                if not fv.empty:
                    fv.loc[:, "matched_pattern"] = selected.get("matched_pattern", selected.get("keyword", ""))
                    render_file_view_row(fv.iloc[0], color_map)
        st.download_button("Download CSV", to_csv_bytes(df), file_name="concordance.csv", mime="text/csv")

elif tool == "Concordance Plot":
    if not query.strip():
        st.warning("Enter a search term first.")
    else:
        df = analyzer.concordance_plot(query=query.strip(), regex=regex, wildcard=wildcard)
        results_store["concordance_plot"] = df
        if df.empty:
            st.info("No matches found.")
        else:
            st.markdown("#### Distribution plot")
            st.caption("Click one point on the plot to open the matching File View position.")
            selected_point = render_distribution_plot(df, key="concordance_plot_select")
            if selected_point is not None:
                st.markdown("#### File View from plot selection")
                fv = analyzer.file_view_at(int(selected_point["doc_id"]), int(selected_point["char_start"]), int(selected_point["char_end"]), context_chars=context_chars)
                if not fv.empty:
                    fv.loc[:, "matched_pattern"] = selected_point.get("matched_pattern", selected_point.get("keyword", ""))
                    render_file_view_row(fv.iloc[0], color_map)
            st.dataframe(apply_table_controls(df, "concordance_plot", sort_default="position_pct"), use_container_width=True, hide_index=True)
        st.download_button("Download CSV", to_csv_bytes(df), file_name="concordance_plot.csv", mime="text/csv")

elif tool == "File View":
    if not query.strip():
        st.warning("Enter a search term first.")
    else:
        df = analyzer.file_view(query=query.strip(), regex=regex, wildcard=wildcard, context_chars=context_chars, max_rows=max_rows)
        results_store["file_view"] = df
        preview = apply_table_controls(df, "file_view", sort_default="char_start")
        if preview.empty:
            st.info("No matches found.")
        else:
            idx = st.selectbox("Choose a row to inspect", list(range(len(preview))), format_func=lambda i: f"{preview.iloc[i]['document']} | {preview.iloc[i]['keyword']} | chars {preview.iloc[i]['char_start']}-{preview.iloc[i]['char_end']}")
            render_file_view_row(preview.iloc[int(idx)], color_map)
            st.dataframe(preview, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", to_csv_bytes(df), file_name="file_view.csv", mime="text/csv")

elif tool == "Clusters / N-Grams":
    n = st.slider("N", 2, 8, 2)
    min_freq = st.number_input("Minimum frequency", 1, 100000, 2)
    df = analyzer.ngrams(n=n, min_freq=int(min_freq), stopwords=stopwords)
    results_store["ngrams"] = df
    st.dataframe(apply_table_controls(df, "ngrams", sort_default="frequency"), use_container_width=True, hide_index=True)
    st.download_button("Download CSV", to_csv_bytes(df), file_name="ngrams.csv", mime="text/csv")

elif tool == "Collocates":
    if not query.strip():
        st.warning("Enter a search term first.")
    else:
        min_freq = st.number_input("Minimum co-occurrence", 1, 100000, 2)
        df = analyzer.collocates(query=query.strip(), window=window, min_freq=int(min_freq), stopwords=stopwords, regex=regex, wildcard=wildcard)
        results_store["collocates"] = df
        st.dataframe(apply_table_controls(df, "collocates", sort_default="co_freq"), use_container_width=True, hide_index=True)
        st.download_button("Download CSV", to_csv_bytes(df), file_name="collocates.csv", mime="text/csv")

elif tool == "Keyword List":
    if reference_analyzer is None:
        st.warning("Upload reference corpus files in the sidebar to use Keyword List.")
    else:
        min_freq = st.number_input("Minimum frequency", 1, 100000, 2)
        df = analyzer.keyword_list(reference_analyzer, min_freq=int(min_freq))
        results_store["keyword_list"] = df
        st.dataframe(apply_table_controls(df, "keyword_list", sort_default="log_likelihood"), use_container_width=True, hide_index=True)
        st.download_button("Download CSV", to_csv_bytes(df), file_name="keyword_list.csv", mime="text/csv")

st.markdown("---")
all_results = build_all_results(
    analyzer,
    reference_analyzer,
    query=query,
    regex=regex,
    wildcard=wildcard,
    stopwords=stopwords,
    window=window,
    context_chars=context_chars,
    max_rows=max_rows,
)

st.download_button(
    "Export workbook with separate sheets + metadata summary",
    data=results_to_excel_bytes(all_results, metadata=metadata_df),
    file_name="antconc_results_full.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
