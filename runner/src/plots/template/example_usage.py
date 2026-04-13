"""
example_usage.py

基于 plot_journal_template 的综合示例脚本。
目的：
1. 将常见科研图形函数集中放入示例用法中；
2. 通过多种版式布局展示模板的适用范围；
3. 运行后统一导出图形，并打包为 zip 文件，便于检查。
"""

from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from plot_journal_template import (
    set_paper_style,
    get_palette,
    get_palette_colors,
    LINESTYLES,
    MARKERS,
    SINGLE_COLUMN,
    DOUBLE_COLUMN,
    TRIPLE_ROW,
    TRIPTYCH,
    add_panel_label,
    export_figure,
)

# =========================================================
# 一、基础设置
# =========================================================
np.random.seed(42)
set_paper_style(base_fontsize=7)

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "coolwarm"

# Linux 风格路径（注释掉）
# OUTPUT_DIR = Path("/mnt/data/plot_journal_example_outputs")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Windows 风格路径
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 二、辅助函数
# =========================================================
def add_colorbar(im, ax=None, label=None, pad=0.02, fraction=0.046):
    cbar = plt.colorbar(im, ax=ax, pad=pad, fraction=fraction)
    if label is not None:
        cbar.set_label(label)
    return cbar


def symmetric_limits(data, center=0.0, robust=False, q=0.98):
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return center - 1.0, center + 1.0
    shifted = np.abs(arr - center)
    radius = np.quantile(shifted, q) if robust else shifted.max()
    return center - radius, center + radius


# ---------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------
def plot_heatmap(
    ax,
    data,
    cmap=SEQUENTIAL_CMAP,
    vmin=None,
    vmax=None,
    colorbar=False,
    colorbar_label=None,
    xticklabels=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    aspect="auto",
    interpolation="nearest",
    rasterized=False,
):
    data = np.asarray(data)
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation=interpolation,
        rasterized=rasterized,
    )
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    cbar = add_colorbar(im, ax=ax, label=colorbar_label) if colorbar else None
    return im, cbar



def plot_diverging_heatmap(
    ax,
    data,
    center=0.0,
    cmap=DIVERGING_CMAP,
    vmin=None,
    vmax=None,
    robust=False,
    q=0.98,
    colorbar=False,
    colorbar_label=None,
    xticklabels=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    aspect="auto",
    interpolation="nearest",
    rasterized=False,
):
    data = np.asarray(data)
    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = symmetric_limits(data, center=center, robust=robust, q=q)
        vmin = vmin_auto if vmin is None else vmin
        vmax = vmax_auto if vmax is None else vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation=interpolation,
        rasterized=rasterized,
    )
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    cbar = add_colorbar(im, ax=ax, label=colorbar_label) if colorbar else None
    return im, cbar



def annotate_heatmap(
    ax,
    data,
    fmt="{:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    fontsize=5.8,
    mask=None,
    use_data_threshold=True,
):
    data = np.asarray(data)
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if threshold is None and use_data_threshold:
        finite = data[np.isfinite(data)]
        threshold = np.nanmedian(finite) if finite.size else 0.0
    elif threshold is None:
        threshold = 0.0
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                continue
            val = data[i, j]
            color = textcolors[int(val > threshold)]
            texts.append(
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )
            )
    return texts



def add_heatmap_group_separators(ax, row_breaks=None, col_breaks=None, color="white", linewidth=1.2):
    if row_breaks is not None:
        for r in row_breaks:
            ax.axhline(r - 0.5, color=color, linewidth=linewidth)
    if col_breaks is not None:
        for c in col_breaks:
            ax.axvline(c - 0.5, color=color, linewidth=linewidth)


# ---------------------------------------------------------
# Scatter / uncertainty / regression
# ---------------------------------------------------------
def plot_errorbar_scatter(
    ax,
    x,
    y,
    xerr=None,
    yerr=None,
    label=None,
    color=None,
    marker="o",
    markerfacecolor="white",
    markeredgewidth=0.8,
    linestyle="None",
    elinewidth=0.8,
    alpha=1.0,
    capsize=2.0,
):
    return ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=marker,
        linestyle=linestyle,
        color=color,
        markerfacecolor=markerfacecolor,
        markeredgewidth=markeredgewidth,
        elinewidth=elinewidth,
        alpha=alpha,
        capsize=capsize,
        label=label,
    )



def plot_line_with_band(
    ax,
    x,
    y,
    yerr=None,
    ymin=None,
    ymax=None,
    color=None,
    label=None,
    linestyle="-",
    linewidth=1.15,
    band_alpha=0.15,
    marker=None,
    markerevery=None,
):
    line = ax.plot(
        x,
        y,
        color=color,
        label=label,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markevery=markerevery,
    )
    if yerr is not None:
        ymin = np.asarray(y) - np.asarray(yerr)
        ymax = np.asarray(y) + np.asarray(yerr)
    if ymin is not None and ymax is not None:
        ax.fill_between(x, ymin, ymax, color=color, alpha=band_alpha, linewidth=0)
    return line



def add_regression_line(
    ax,
    x,
    y,
    color=None,
    linestyle="--",
    linewidth=1.0,
    label=None,
    annotate=True,
    annotation_loc=(0.03, 0.95),
    annotation_fmt="r = {:.2f}",
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("Need at least two valid points for regression.")
    slope, intercept = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = slope * xfit + intercept
    ax.plot(xfit, yfit, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    r = np.corrcoef(x, y)[0, 1]
    if annotate:
        ax.text(
            annotation_loc[0],
            annotation_loc[1],
            annotation_fmt.format(r),
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    return slope, intercept, r


# ---------------------------------------------------------
# Bars / distributions / statistical annotations
# ---------------------------------------------------------
def plot_grouped_bars(
    ax,
    values,
    errors=None,
    category_labels=None,
    group_labels=None,
    palette_name="default",
    width=None,
    group_gap=0.0,
):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array with shape (n_groups, n_categories)")
    n_groups, n_categories = values.shape
    pal = get_palette(palette_name)
    colors = pal["colors"]
    x = np.arange(n_categories)
    width = 0.8 / n_groups if width is None else width
    handles = []
    for i in range(n_groups):
        xpos = x + (i - (n_groups - 1) / 2) * width * (1 + group_gap)
        yerr = None if errors is None else np.asarray(errors)[i]
        bars = ax.bar(
            xpos,
            values[i],
            width=width,
            yerr=yerr,
            color=colors[i % len(colors)],
            error_kw=dict(elinewidth=0.7, capsize=2, ecolor=pal["accent"]),
            label=None if group_labels is None else group_labels[i],
        )
        handles.append(bars)
    ax.set_xticks(x)
    if category_labels is not None:
        ax.set_xticklabels(category_labels)
    return handles



def plot_boxplot(
    ax,
    data,
    labels=None,
    palette_name="default",
    showfliers=False,
    widths=0.58,
    box_alpha=0.82,
    patch_artist=True,
):
    pal = get_palette(palette_name)
    colors = pal["colors"]
    bp = ax.boxplot(
        data,
        patch_artist=patch_artist,
        widths=widths,
        showfliers=showfliers,
        medianprops=dict(color=pal["accent"], linewidth=1.0),
        whiskerprops=dict(color=pal["gray"], linewidth=0.8),
        capprops=dict(color=pal["gray"], linewidth=0.8),
        boxprops=dict(linewidth=0.8, color=pal["gray"]),
        flierprops=dict(
            marker="o",
            markersize=2.0,
            markerfacecolor="white",
            markeredgecolor=pal["gray"],
            alpha=0.8,
        ),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(box_alpha)
    if labels is not None:
        ax.set_xticklabels(labels)
    return bp



def plot_violin(
    ax,
    data,
    labels=None,
    palette_name="default",
    widths=0.9,
    alpha=0.82,
    showmeans=False,
    showmedians=True,
    showextrema=True,
):
    pal = get_palette(palette_name)
    colors = pal["colors"]
    vp = ax.violinplot(data, widths=widths, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor(pal["gray"])
        body.set_alpha(alpha)
    for key in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if key in vp:
            vp[key].set_color(pal["gray"])
            vp[key].set_linewidth(0.8)
    if labels is not None:
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    return vp



def format_pvalue(p, style="stars"):
    if style == "stars":
        if p < 1e-4:
            return "****"
        elif p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 5e-2:
            return "*"
        else:
            return "ns"
    elif style == "scientific":
        return f"p = {p:.2e}"
    elif style == "plain":
        return f"p = {p:.4f}"
    raise ValueError(f"Unknown style: {style}")



def add_significance_bar(
    ax,
    x1,
    x2,
    y,
    h=0.02,
    text=None,
    pvalue=None,
    style="stars",
    fontsize=6.5,
    linewidth=0.8,
    color="black",
    text_offset=0.005,
):
    if text is None and pvalue is not None:
        text = format_pvalue(pvalue, style=style)
    if text is None:
        text = "*"
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=linewidth, c=color)
    ax.text((x1 + x2) / 2, y + h + text_offset, text, ha="center", va="bottom", fontsize=fontsize, color=color)
    return text



def add_pvalue_text(ax, x, y, pvalue=None, text=None, style="scientific", ha="left", va="bottom", fontsize=6.5):
    if text is None and pvalue is not None:
        text = format_pvalue(pvalue, style=style)
    if text is None:
        raise ValueError("Provide either pvalue or text.")
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize)


# ---------------------------------------------------------
# Forest / benchmark
# ---------------------------------------------------------
def plot_forest(
    ax,
    centers,
    errors,
    row_labels=None,
    series_labels=None,
    palette_name="default",
    markerfacecolor="white",
    reference_line=None,
    x_label=None,
    y_label=None,
):
    centers = np.asarray(centers, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if centers.ndim != 2:
        raise ValueError("centers must be a 2D array with shape (n_series, n_rows)")
    if errors.shape != centers.shape:
        raise ValueError("errors must match centers shape")
    n_series, n_rows = centers.shape
    pal = get_palette(palette_name)
    colors = pal["colors"]
    ypos = np.arange(n_rows)[::-1]
    offsets = np.linspace(-0.26, 0.26, n_series)
    for i in range(n_series):
        ax.errorbar(
            centers[i],
            ypos + offsets[i],
            xerr=errors[i],
            fmt=MARKERS[i % len(MARKERS)],
            linestyle="None",
            color=colors[i % len(colors)],
            markerfacecolor=markerfacecolor,
            markeredgewidth=0.8,
            label=None if series_labels is None else series_labels[i],
            elinewidth=0.8,
        )
    ax.set_yticks(ypos)
    if row_labels is not None:
        ax.set_yticklabels(row_labels)
    if reference_line is not None:
        ax.axvline(reference_line, color=pal["gray"], linewidth=0.8, linestyle="--")
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    return ax


# ---------------------------------------------------------
# ML / biostat publication plots
# ---------------------------------------------------------
def plot_volcano(
    ax,
    log2fc,
    pvalues,
    labels=None,
    palette_name="default",
    fc_thresh=1.0,
    p_thresh=0.05,
    s=10,
    alpha=0.75,
    label_top_n=0,
    up_label="Up",
    down_label="Down",
    other_label="NS",
):
    log2fc = np.asarray(log2fc, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    neglogp = -np.log10(np.clip(pvalues, 1e-300, None))
    pal = get_palette(palette_name)
    colors = pal["colors"]

    up = (log2fc >= fc_thresh) & (pvalues < p_thresh)
    down = (log2fc <= -fc_thresh) & (pvalues < p_thresh)
    ns = ~(up | down)

    ax.scatter(log2fc[ns], neglogp[ns], s=s, color=pal["light_gray"], alpha=alpha, edgecolors="none", label=other_label)
    ax.scatter(log2fc[up], neglogp[up], s=s, color=colors[0], alpha=alpha, edgecolors="none", label=up_label)
    ax.scatter(log2fc[down], neglogp[down], s=s, color=colors[4 % len(colors)], alpha=alpha, edgecolors="none", label=down_label)

    ax.axvline(fc_thresh, color=pal["gray"], linestyle="--", linewidth=0.8)
    ax.axvline(-fc_thresh, color=pal["gray"], linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(p_thresh), color=pal["gray"], linestyle="--", linewidth=0.8)

    if labels is not None and label_top_n > 0:
        labels = np.asarray(labels)
        score = neglogp * np.abs(log2fc)
        idx = np.argsort(score)[-label_top_n:]
        for i in idx:
            ax.text(log2fc[i], neglogp[i], str(labels[i]), fontsize=5.8, ha="left", va="bottom")

    ax.set_xlabel("log$_2$ fold change")
    ax.set_ylabel("-log$_{10}$(p-value)")
    ax.legend(loc="upper right")
    return {"up_mask": up, "down_mask": down, "ns_mask": ns}



def trapezoid_auc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    # return np.trapezoid(y[order], x[order])
    return np.trapz(y[order], x[order])



def plot_roc_curve(
    ax,
    fpr,
    tpr,
    label=None,
    color=None,
    linestyle="-",
    linewidth=1.2,
    diagonal=True,
    annotate_auc=True,
    auc_loc=(0.60, 0.08),
    auc_fmt="AUC = {:.3f}",
):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    auc = trapezoid_auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    if diagonal:
        ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=0.8)
    if annotate_auc:
        ax.text(auc_loc[0], auc_loc[1], auc_fmt.format(auc), transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    return auc



def plot_pr_curve(
    ax,
    recall,
    precision,
    label=None,
    color=None,
    linestyle="-",
    linewidth=1.2,
    baseline=None,
    annotate_auc=True,
    auc_loc=(0.60, 0.08),
    auc_fmt="AUPRC = {:.3f}",
):
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    auc = trapezoid_auc(recall, precision)
    ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    if baseline is not None:
        ax.axhline(baseline, color="0.7", linestyle="--", linewidth=0.8)
    if annotate_auc:
        ax.text(auc_loc[0], auc_loc[1], auc_fmt.format(auc), transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return auc



def kaplan_meier_estimate(times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_event_times = np.unique(times[events == 1])
    if unique_event_times.size == 0:
        return np.array([0.0]), np.array([1.0])

    xs = [0.0]
    ys = [1.0]
    n_at_risk = len(times)

    for t in unique_event_times:
        d_i = np.sum((times == t) & (events == 1))
        c_i = np.sum((times == t) & (events == 0))
        xs.extend([t, t])
        ys.extend([ys[-1], ys[-1] * (1.0 - d_i / n_at_risk)])
        n_at_risk -= (d_i + c_i)

    return np.array(xs), np.array(ys)



def plot_kaplan_meier(
    ax,
    times,
    events,
    label=None,
    color=None,
    linewidth=1.2,
    show_censors=True,
    censor_marker="|",
    censor_size=18,
):
    x, y = kaplan_meier_estimate(times, events)
    ax.step(x, y, where="post", color=color, linewidth=linewidth, label=label)
    if show_censors:
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)
        censor_times = times[events == 0]
        if censor_times.size > 0:
            for ct in censor_times:
                xv, yv = kaplan_meier_estimate(times[times <= ct], events[times <= ct])
                ax.plot(ct, yv[-1], marker=censor_marker, color=color, markersize=censor_size**0.5, linestyle="None")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    return x, y


# =========================================================
# 三、示例数据
# =========================================================
colors = get_palette_colors("default")
pal = get_palette("default")

x = np.linspace(0, 10, 300)
ys = [
    0.25 + 0.10 * x,
    0.40 + 0.45 * np.exp(-0.5 * ((x - 3.0) / 1.15) ** 2),
    0.55 + 0.22 * np.sin(np.pi * x / 10),
]

heat = np.random.gamma(shape=2.0, scale=1.0, size=(10, 12))
delta = np.random.normal(loc=0.0, scale=1.0, size=(8, 10))

scatter_x = np.linspace(0.5, 8.0, 18)
scatter_y = 1.2 + 0.65 * scatter_x + np.random.normal(scale=0.9, size=scatter_x.size)
scatter_xerr = 0.12 + 0.08 * np.random.rand(scatter_x.size)
scatter_yerr = 0.25 + 0.15 * np.random.rand(scatter_x.size)

line_x = np.linspace(0, 20, 50)
line_y = 0.4 + 0.03 * line_x + 0.12 * np.sin(line_x / 3)
line_err = 0.08 + 0.02 * np.cos(line_x / 4)

bar_values = np.array([
    [0.82, 0.74, 0.69, 0.77],
    [0.88, 0.79, 0.73, 0.81],
    [0.91, 0.84, 0.76, 0.86],
])
bar_errors = np.array([
    [0.03, 0.02, 0.03, 0.02],
    [0.02, 0.03, 0.02, 0.03],
    [0.02, 0.02, 0.03, 0.02],
])

box_data = [
    np.random.normal(0.85, 0.08, 70),
    np.random.normal(0.72, 0.10, 70),
    np.random.normal(0.63, 0.12, 70),
    np.random.normal(0.78, 0.09, 70),
]
violin_data = [
    np.random.gamma(3.0, 0.35, 100),
    np.random.gamma(4.0, 0.28, 100),
    np.random.gamma(5.0, 0.22, 100),
]

forest_centers = np.array([
    [0.92, 1.10, 0.86, 1.05, 0.98],
    [1.02, 1.18, 0.91, 1.14, 1.04],
    [0.96, 1.06, 0.88, 1.08, 1.00],
])
forest_errors = np.array([
    [0.08, 0.10, 0.06, 0.09, 0.07],
    [0.07, 0.09, 0.05, 0.08, 0.06],
    [0.06, 0.08, 0.05, 0.07, 0.05],
])

n_genes = 300
log2fc = np.random.normal(0, 1.2, n_genes)
pvalues = np.clip(np.random.beta(0.8, 5.0, n_genes), 1e-6, 1.0)
labels = np.array([f"G{i+1}" for i in range(n_genes)])

fpr = np.linspace(0, 1, 200)
tpr_model_a = 1 - (1 - fpr) ** 3.2
tpr_model_b = 1 - (1 - fpr) ** 2.2
recall = np.linspace(0, 1, 200)
precision_model_a = 0.92 - 0.55 * recall ** 1.25
precision_model_b = 0.82 - 0.48 * recall ** 1.10
precision_model_a = np.clip(precision_model_a, 0, 1)
precision_model_b = np.clip(precision_model_b, 0, 1)

km_times_a = np.random.exponential(scale=12.0, size=70)
km_events_a = np.random.binomial(1, 0.72, size=70)
km_times_b = np.random.exponential(scale=18.0, size=70)
km_events_b = np.random.binomial(1, 0.65, size=70)


# =========================================================
# 四、导出示例图
# =========================================================
# 示例 1：单栏单图
fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
for i, y in enumerate(ys):
    plot_line_with_band(
        ax,
        x,
        y,
        yerr=0.02 + 0.01 * np.sin(x / 3 + i),
        color=colors[i],
        label=f"Series {i + 1}",
        linestyle=LINESTYLES[i],
    )
ax.set_xlabel("Time")
ax.set_ylabel("Response")
ax.legend(loc="upper left")
add_panel_label(ax, index=0, style="upper")
export_figure(fig, "example_01_single_column_line", outdir=OUTPUT_DIR)
plt.close(fig)

# 示例 2：双栏 2x2 综合版式
fig, axs = plt.subplots(2, 2, figsize=DOUBLE_COLUMN)

for i, y in enumerate(ys):
    axs[0, 0].plot(x, y, color=colors[i], linestyle=LINESTYLES[i], label=f"S{i + 1}")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Signal")
axs[0, 0].legend(loc="upper left")

plot_errorbar_scatter(axs[0, 1], scatter_x, scatter_y, xerr=scatter_xerr, yerr=scatter_yerr, color=colors[0], label="Samples")
add_regression_line(axs[0, 1], scatter_x, scatter_y, color=colors[3])
axs[0, 1].set_xlabel("Predictor")
axs[0, 1].set_ylabel("Response")

plot_grouped_bars(
    axs[1, 0],
    bar_values,
    errors=bar_errors,
    category_labels=["Task 1", "Task 2", "Task 3", "Task 4"],
    group_labels=["Method A", "Method B", "Method C"],
)
axs[1, 0].set_ylabel("Score")
axs[1, 0].legend(loc="upper left")

plot_boxplot(axs[1, 1], box_data, labels=["G1", "G2", "G3", "G4"])
axs[1, 1].set_ylabel("Distribution")
add_significance_bar(axs[1, 1], 1, 2, y=1.05, h=0.03, pvalue=0.003)
add_pvalue_text(axs[1, 1], 3.2, 1.08, pvalue=0.017, style="plain")

for k, ax in enumerate(axs.ravel()):
    add_panel_label(ax, index=k, style="paren")
fig.subplots_adjust(hspace=0.55, wspace=0.42)
export_figure(fig, "example_02_double_column_2x2", outdir=OUTPUT_DIR)
plt.close(fig)

# 示例 3：三联横排，突出热图/差异热图/森林图
fig, axs = plt.subplots(1, 3, figsize=TRIPTYCH)
plot_heatmap(
    axs[0],
    heat,
    colorbar=True,
    colorbar_label="Intensity",
    xlabel="Feature",
    ylabel="Sample",
    title="Sequential heatmap",
)
add_heatmap_group_separators(axs[0], row_breaks=[3, 7], col_breaks=[4, 8], color="white")

im, _ = plot_diverging_heatmap(
    axs[1],
    delta,
    center=0.0,
    robust=True,
    colorbar=True,
    colorbar_label="Δ value",
    xlabel="Condition",
    ylabel="Gene set",
    title="Diverging heatmap",
)
annotate_heatmap(axs[1], delta, fmt="{:.1f}", fontsize=5.0)

plot_forest(
    axs[2],
    forest_centers,
    forest_errors,
    row_labels=["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5"],
    series_labels=["M1", "M2", "M3"],
    reference_line=1.0,
    x_label="Effect size",
)
axs[2].legend(loc="lower right")

for k, ax in enumerate(axs.ravel()):
    add_panel_label(ax, index=k, style="upper")
fig.subplots_adjust(wspace=0.55)
export_figure(fig, "example_03_triptych_heatmap_forest", outdir=OUTPUT_DIR)
plt.close(fig)

# 示例 4：双栏机器学习/生信常用版式
fig, axs = plt.subplots(2, 2, figsize=DOUBLE_COLUMN)
plot_volcano(axs[0, 0], log2fc, pvalues, labels=labels, label_top_n=8)

plot_roc_curve(axs[0, 1], fpr, tpr_model_a, label="Model A", color=colors[0])
plot_roc_curve(axs[0, 1], fpr, tpr_model_b, label="Model B", color=colors[1], annotate_auc=False)
axs[0, 1].legend(loc="lower right")

plot_pr_curve(axs[1, 0], recall, precision_model_a, label="Model A", color=colors[0], baseline=0.28)
plot_pr_curve(axs[1, 0], recall, precision_model_b, label="Model B", color=colors[1], annotate_auc=False)
axs[1, 0].legend(loc="lower left")

plot_kaplan_meier(axs[1, 1], km_times_a, km_events_a, label="Group A", color=colors[0])
plot_kaplan_meier(axs[1, 1], km_times_b, km_events_b, label="Group B", color=colors[3])
axs[1, 1].legend(loc="upper right")

for k, ax in enumerate(axs.ravel()):
    add_panel_label(ax, index=k, style="paren_upper")
fig.subplots_adjust(hspace=0.52, wspace=0.42)
export_figure(fig, "example_04_ml_biostat_panels", outdir=OUTPUT_DIR)
plt.close(fig)

# 示例 5：双栏不对称 mosaic 版式
fig = plt.figure(figsize=DOUBLE_COLUMN)
mosaic = fig.subplot_mosaic(
    [["A", "B"], ["C", "B"]],
    gridspec_kw={"width_ratios": [1.0, 1.12], "height_ratios": [1.0, 1.0]},
)

plot_violin(mosaic["A"], violin_data, labels=["State 1", "State 2", "State 3"])
mosaic["A"].set_ylabel("Expression")

plot_diverging_heatmap(
    mosaic["B"],
    np.random.normal(0, 1, (14, 8)),
    center=0.0,
    robust=True,
    colorbar=True,
    colorbar_label="Z-score",
    xlabel="Latent factor",
    ylabel="Feature block",
)

for i in range(3):
    y = 0.15 * i + np.exp(-((x - (2.5 + 2.0 * i)) ** 2) / (2 * (1.0 + 0.15 * i) ** 2))
    mosaic["C"].plot(x, y, color=colors[i], linestyle=LINESTYLES[i], label=f"Condition {i + 1}")
mosaic["C"].set_xlabel("Position")
mosaic["C"].set_ylabel("Density")
mosaic["C"].legend(loc="upper right")

for idx, key in enumerate(["A", "B", "C"]):
    add_panel_label(mosaic[key], index=idx, style="upper")
fig.subplots_adjust(hspace=0.45, wspace=0.35)
export_figure(fig, "example_05_asymmetric_mosaic", outdir=OUTPUT_DIR)
plt.close(fig)

# 示例 6：横向窄图，适合补充材料/benchmark
fig, ax = plt.subplots(figsize=TRIPLE_ROW)
for i in range(4):
    y = 0.55 + 0.08 * i + 0.10 * np.sin(x / (1.7 + 0.2 * i))
    ax.plot(x, y, color=colors[i], linestyle=LINESTYLES[i], label=f"Run {i + 1}")
ax.set_xlabel("Iteration")
ax.set_ylabel("Metric")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
add_panel_label(ax, index=0, style="upper")
export_figure(fig, "example_06_triple_row_benchmark", outdir=OUTPUT_DIR)
plt.close(fig)

# 打包输出文件
# Linux 风格路径（注释掉）
# zip_base = Path("/mnt/data/plot_journal_examples_bundle")
# if zip_base.with_suffix(".zip").exists():
#     zip_base.with_suffix(".zip").unlink()
# shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_DIR)

# Windows 风格路径
zip_base = Path("plot_journal_examples_bundle")
if zip_base.with_suffix(".zip").exists():
    zip_base.with_suffix(".zip").unlink()
shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_DIR)

print(f"Generated outputs in: {OUTPUT_DIR}")
print(f"Created zip archive: {zip_base.with_suffix('.zip')}")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(p.name)
