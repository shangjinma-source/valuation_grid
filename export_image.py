"""
export_image.py - 服务端批量导出实时估值板块图片
完全复刻 demo.html 中 exportSectorImage() 的 Canvas 绘制逻辑
"""
import base64
import io
from datetime import datetime
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from valuation.core import load_state, calculate_valuation_batch


# === 字体 ===
# 服务端无 macOS 系统字体，使用 PIL 默认字体（等宽）
# 若需更好效果可替换为 .ttf 字体文件路径
def _get_font(size: int, bold: bool = False):
    """尝试加载系统中文字体，fallback 到 PIL 默认字体"""
    font_paths = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑粗体
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    # fallback
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


# === 颜色常量（与前端一致） ===
COLOR_BG       = "#ffffff"
COLOR_HEAD_BG  = "#f1f5f9"
COLOR_ROW_ALT  = "#f8fafc"
COLOR_TITLE    = "#1e293b"
COLOR_TIME     = "#9ca3af"
COLOR_TH       = "#64748b"
COLOR_CODE     = "#6366f1"
COLOR_NAME     = "#334155"
COLOR_UP       = "#ef4444"
COLOR_DOWN     = "#22c55e"
COLOR_FLAT     = "#64748b"
COLOR_NAV_DATE = "#f59e0b"


def _change_text(val: Optional[float]) -> tuple:
    """格式化涨跌幅文本和颜色（与前端一致）"""
    if val is None:
        return "--", COLOR_FLAT
    txt = f"{'+' if val >= 0 else ''}{val:.2f}%"
    clr = COLOR_UP if val > 0 else (COLOR_DOWN if val < 0 else COLOR_FLAT)
    return txt, clr


def render_sector_image(
    sector_name: str,
    funds: list,
    valuations: dict,
    scale: int = 2,
) -> bytes:
    """
    渲染单个板块的估值图片（PNG bytes）。
    参数与前端 exportSectorImage 一一对应：
      - funds: [{code, alias}, ...]
      - valuations: {code: {estimation_change, week_change, month_change, ...}, ...}
    """
    # --- 筛选 + 排序（与前端完全一致）---
    sorted_funds = []
    for f in funds:
        code = f.get("code", "")
        v = valuations.get(code)
        if not v:
            continue
        # 净值来源直接输出
        if v.get("_source") == "nav":
            sorted_funds.append(f)
            continue
        # 盘中估值：校准后置信度 >= 0.50
        conf = v.get("calibrated_confidence") or v.get("confidence", 0)
        if conf >= 0.50:
            sorted_funds.append(f)

    sorted_funds.sort(
        key=lambda f: valuations.get(f["code"], {}).get("estimation_change") or -999,
        reverse=True,
    )

    if not sorted_funds:
        return b""

    # --- 布局常量（与前端一致）---
    P = 16
    row_h = 26
    head_h = 36
    th_h = 22
    W = 560
    H = head_h + th_h + len(sorted_funds) * row_h + P

    col_code_l  = P
    col_name_l  = P + 58
    col_change_r = W - P - 155
    col_5day_r   = W - P - 78
    col_20day_r  = W - P

    # --- 创建高清画布 ---
    img = Image.new("RGB", (W * scale, H * scale), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # 字体（按 scale 缩放）
    font_title  = _get_font(14 * scale, bold=True)
    font_time   = _get_font(11 * scale)
    font_th     = _get_font(10 * scale)
    font_code   = _get_font(11 * scale)
    font_name   = _get_font(11 * scale)
    font_val    = _get_font(12 * scale, bold=True)
    font_navdt  = _get_font(9 * scale)

    S = scale  # 简写

    # --- 标题行 ---
    draw.text((P * S, P * S), sector_name, fill=COLOR_TITLE, font=font_title)

    time_str = datetime.now().strftime("%m/%d %H:%M")
    tw_time = draw.textlength(time_str, font=font_time)
    draw.text(((W - P) * S - tw_time, P * S), time_str, fill=COLOR_TIME, font=font_time)

    # --- 表头背景 ---
    draw.rectangle(
        [0, head_h * S, W * S, (head_h + th_h) * S],
        fill=COLOR_HEAD_BG,
    )

    # 表头文字
    th_y = (head_h + 4) * S  # 微调垂直居中
    draw.text((col_code_l * S, th_y), "代码", fill=COLOR_TH, font=font_th)
    draw.text((col_name_l * S, th_y), "基金名称", fill=COLOR_TH, font=font_th)

    for label, right_x in [("估值涨幅", col_change_r), ("近5日", col_5day_r), ("近20日", col_20day_r)]:
        tw = draw.textlength(label, font=font_th)
        draw.text((right_x * S - tw, th_y), label, fill=COLOR_TH, font=font_th)

    # --- 数据行 ---
    today_str = datetime.now().strftime("%Y-%m-%d")

    for i, fund in enumerate(sorted_funds):
        code = fund["code"]
        val = valuations.get(code, {})
        y = head_h + th_h + i * row_h
        text_y = (y + 6) * S  # 行内文字基线（微调）

        # 斑马纹
        if i % 2 == 1:
            draw.rectangle([0, y * S, W * S, (y + row_h) * S], fill=COLOR_ROW_ALT)

        # 代码
        draw.text((col_code_l * S, text_y), code, fill=COLOR_CODE, font=font_code)

        # 名称（截断）
        name = fund.get("alias") or val.get("fund_name") or ""
        if len(name) > 14:
            name = name[:14] + "..."
        draw.text((col_name_l * S, text_y), name, fill=COLOR_NAME, font=font_name)

        # 估值涨幅（右对齐）
        chg_txt, chg_clr = _change_text(val.get("estimation_change"))
        tw = draw.textlength(chg_txt, font=font_val)
        draw.text((col_change_r * S - tw, text_y), chg_txt, fill=chg_clr, font=font_val)

        # 净值日期标注
        if val.get("_source") == "nav" and val.get("_nav_date"):
            nav_date = val["_nav_date"]
            if nav_date != today_str:
                md = nav_date[5:].lstrip("0").replace("-0", "/").replace("-", "/")
                draw.text(((col_change_r + 2) * S, text_y), md, fill=COLOR_NAV_DATE, font=font_navdt)

        # 近5日（右对齐）
        w_txt, w_clr = _change_text(val.get("week_change"))
        tw = draw.textlength(w_txt, font=font_val)
        draw.text((col_5day_r * S - tw, text_y), w_txt, fill=w_clr, font=font_val)

        # 近20日（右对齐）
        m_txt, m_clr = _change_text(val.get("month_change"))
        tw = draw.textlength(m_txt, font=font_val)
        draw.text((col_20day_r * S - tw, text_y), m_txt, fill=m_clr, font=font_val)

    # --- 导出 PNG ---
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def export_all_sector_images() -> list:
    """
    批量导出所有板块估值图片。
    返回: [{"sector": name, "filename": ..., "image_base64": ...}, ...]
    功能等同于前端 exportAllSectorImages()。
    """
    state = load_state()
    sectors = state.get("sectors", [])
    if not sectors:
        return []

    # 收集所有基金代码，一次性批量估值
    all_codes = []
    for sector in sectors:
        for fund in sector.get("funds", []):
            code = fund.get("code", "")
            if code and code not in all_codes:
                all_codes.append(code)

    if not all_codes:
        return []

    # 批量估值（与前端 refreshAll 逻辑一致）
    batch_results = calculate_valuation_batch(all_codes)
    valuations = {r["fund_code"]: r for r in batch_results}

    # 逐板块渲染
    results = []
    time_str = datetime.now().strftime("%m%d_%H%M")

    for sector in sectors:
        funds = sector.get("funds", [])
        if not funds:
            continue

        name = sector.get("name", "未命名")
        png_bytes = render_sector_image(name, funds, valuations)
        if not png_bytes:
            continue

        filename = f"{name}_{time_str}.png"
        results.append({
            "sector": name,
            "filename": filename,
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
        })

    return results