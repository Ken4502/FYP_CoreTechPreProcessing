import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import io

import importlib
import core.methods
import core.normalization
import core.scraper

importlib.reload(core.methods)
from core.methods import PlantPreprocessors
from core.normalization import check_blur
from core.scraper import search_plant_images_online, fetch_image_from_url

# Set page configuration
st.set_page_config(
    page_title="Plant Preprocessing Studio - Web Scraper & Segmentation",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern clean aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 1.2rem;
    }
    .method-card {
        background-color: #f9fbe7;
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 16px;
    }
    .method-card-disease {
        background-color: #fff8e1;
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid #f57f17;
        margin-bottom: 16px;
    }
    .scraper-box {
        background: #f1f8e9;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #c8e6c9;
        margin-bottom: 20px;
    }
    .badge-wiki {
        background-color: #e8f5e9;
        color: #2e7d32;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-inat {
        background-color: #e3f2fd;
        color: #1565c0;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_img_rgb" not in st.session_state:
    st.session_state.current_img_rgb = None
if "current_img_title" not in st.session_state:
    st.session_state.current_img_title = "None"
if "scraped_results" not in st.session_state:
    st.session_state.scraped_results = []
if "selected_img_id" not in st.session_state:
    st.session_state.selected_img_id = None
if "search_query" not in st.session_state:
    st.session_state.search_query = "Monstera deliciosa leaf"
if "last_preset_choice" not in st.session_state:
    st.session_state.last_preset_choice = None

# Preprocessor instance
preprocessor = PlantPreprocessors()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("🌿 Image Input Source")

    input_source = st.radio(
        "Select Source:",
        [
            "🌐 Scrape / Search Web Images",
            "🔗 Direct Image URL",
            "📁 Preset Test Sample",
            "📤 Upload Your Own Image"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🧠 AI Segmentation Engine")
    engine_choice = st.selectbox(
        "Model Architecture:",
        [
            "BiRefNet (Recommended - Complex Backgrounds & Outdoor Plants)",
            "U2Net (Fast - Cut Leaves on Plain Background)"
        ],
        index=0,
        help="BiRefNet accurately isolates the foreground plant from other plants in the background."
    )
    model_name = "birefnet-general" if "BiRefNet" in engine_choice else "u2net"

    st.markdown("---")
    st.markdown("### ✂️ Leaf Shape & Cropping Controls")
    preserve_leaf_shape = st.checkbox(
        "Preserve Leaf Apex & Margins",
        value=True,
        help="Guarantees delicate pointed leaf tips, serrations, and margins are not clipped or eroded."
    )
    padding_val = st.slider(
        "Crop Margin Padding (%):",
        min_value=5,
        max_value=35,
        value=15,
        step=1,
        help="Adds safe breathing room around the plant so that outward-reaching leaf tips are never sliced off."
    )
    padding_ratio = padding_val / 100.0

    filter_pot_soil = st.checkbox(
        "Filter Out Pot & Soil", 
        value=True, 
        help="Discards non-plant containers (plastic/terracotta pots) and soil beds so only pure plant foliage & stems are passed downstream."
    )

    st.markdown("---")
    st.markdown("### 🔍 Foliage Segmentation Scope")
    scope_choice = st.radio(
        "Foliage Scope:",
        [
            "🌱 Prominent Leaf Focus (Recommended - High-Res & Disease)",
            "🌳 Full Bush Canopy (Broad Cluster - All Connected Foliage)"
        ],
        index=0,
        help="Prominent Leaf Focus removes distant stray leaf scraps and zooms in directly on the primary foreground leaves to maximize pixel density on veins, margins, and disease spots."
    )
    foliage_scope = "salient_focus" if "Prominent" in scope_choice else "full_canopy"

    st.markdown("---")
    st.markdown("### 🎨 Background Fill")
    bg_style_choice = st.selectbox(
        "Background Color for Cropped Plant:",
        ["Neutral Gray (128, 128, 128)", "Pure Black (0, 0, 0)", "Pure White (255, 255, 255)"],
        index=0,
        help="Neutral gray (128) is recommended for machine learning models to prevent convolutional boundary bias."
    )

    if "White" in bg_style_choice:
        bg_rgb = (255, 255, 255)
    elif "Black" in bg_style_choice:
        bg_rgb = (0, 0, 0)
    else:
        bg_rgb = (128, 128, 128)

# -------------------------------------------------------------
# MAIN VIEW: TITLE & INPUT HANDLERS
# -------------------------------------------------------------
st.markdown('<div class="main-title">🌿 Plant Image Preprocessing Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Scrape plant & disease photos online or load local samples, then generate dual precision crops for <strong>Species ID</strong> and <strong>Disease Diagnosis</strong>.</div>', unsafe_allow_html=True)

# --- 1. WEB SCRAPER MODE ---
if input_source == "🌐 Scrape / Search Web Images":
    with st.container():
        st.markdown("""
        <div class="scraper-box">
            <h4 style="margin: 0 0 6px 0; color: #1b5e20;">🌐 Search & Scrape Plant Images Online</h4>
            <p style="margin: 0 0 12px 0; color: #37474f; font-size: 0.95rem;">
                Fetches real botanical and plant pathology photos directly from <strong>Wikimedia Commons</strong> & <strong>iNaturalist</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Quick Preset Queries:**")
        quick_chips = [
            ("🌿 Monstera deliciosa", "Monstera deliciosa leaf"),
            ("🍅 Tomato Late Blight", "Late blight on tomato leaf"),
            ("🍏 Apple Scab", "Apple scab leaf"),
            ("🪴 Dieffenbachia", "Dieffenbachia leaf"),
            ("🍃 Ficus Lyrata", "Ficus lyrata leaf"),
            ("🍇 Grape Black Rot", "Grape black rot leaf"),
            ("☕ Coffee Leaf Rust", "Coffee leaf rust"),
            ("🌵 Snake Plant", "Sansevieria trifasciata plant")
        ]

        chip_cols = st.columns(4)
        for i, (chip_label, chip_q) in enumerate(quick_chips):
            col_target = chip_cols[i % 4]
            with col_target:
                if st.button(chip_label, key=f"chip_btn_{i}", use_container_width=True):
                    st.session_state.search_query = chip_q
                    with st.spinner(f"Scraping online images for '{chip_q}'..."):
                        results = search_plant_images_online(chip_q, limit=8)
                        st.session_state.scraped_results = results
                        if results:
                            # Auto-fetch the first image
                            first_item = results[0]
                            dl_url = first_item.get('thumb_url') or first_item.get('full_url')
                            img, err = fetch_image_from_url(dl_url)
                            if img is not None:
                                st.session_state.current_img_rgb = img
                                st.session_state.current_img_title = first_item['title']
                                st.session_state.selected_img_id = first_item['id']
                    st.rerun()

        st.markdown("---")
        # Search input bar
        search_c1, search_c2 = st.columns([5, 1])
        with search_c1:
            q_input = st.text_input(
                "Custom Query:",
                value=st.session_state.search_query,
                placeholder="e.g. Monstera deliciosa, Tomato late blight, Apple scab...",
                label_visibility="collapsed"
            )
        with search_c2:
            search_exec = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_exec or (not st.session_state.scraped_results and len(st.session_state.search_query) > 1):
            with st.spinner(f"Scraping plant images for '{q_input}'..."):
                st.session_state.search_query = q_input
                results = search_plant_images_online(q_input, limit=8)
                st.session_state.scraped_results = results
                if results and st.session_state.current_img_rgb is None:
                    first_item = results[0]
                    dl_url = first_item.get('thumb_url') or first_item.get('full_url')
                    img, err = fetch_image_from_url(dl_url)
                    if img is not None:
                        st.session_state.current_img_rgb = img
                        st.session_state.current_img_title = first_item['title']
                        st.session_state.selected_img_id = first_item['id']

        # Render Gallery of Results
        if st.session_state.scraped_results:
            st.markdown(f"**Found {len(st.session_state.scraped_results)} images for:** `{st.session_state.search_query}` (Click **⚡ Process This** to segment)")
            grid_cols = st.columns(4)
            for idx, item in enumerate(st.session_state.scraped_results):
                target_col = grid_cols[idx % 4]
                with target_col:
                    is_active = (st.session_state.selected_img_id == item['id'])
                    st.image(item['thumb_url'], use_container_width=True)
                    badge_cls = "badge-wiki" if "Wikimedia" in item.get('source', '') else "badge-inat"
                    st.markdown(f"<span class='{badge_cls}'>{item.get('source', '')}</span>", unsafe_allow_html=True)
                    st.caption(f"**{item['title'][:32]}**")
                    if is_active:
                        st.success("✓ Currently Active")
                    else:
                        if st.button("⚡ Process This", key=f"sel_item_{item['id']}", use_container_width=True):
                            with st.spinner("Downloading image from web..."):
                                dl_url = item.get('thumb_url') or item.get('full_url')
                                img, err = fetch_image_from_url(dl_url)
                                if img is not None:
                                    st.session_state.current_img_rgb = img
                                    st.session_state.current_img_title = item['title']
                                    st.session_state.selected_img_id = item['id']
                                    st.rerun()
                                else:
                                    st.error(err)
        else:
            st.warning("No images found for this query. Try a general botanical or plant disease name above.")

# --- 2. DIRECT IMAGE URL MODE ---
elif input_source == "🔗 Direct Image URL":
    st.markdown("### 🔗 Load Image Directly from Web URL")
    url_input_col1, url_input_col2 = st.columns([5, 1])
    with url_input_col1:
        custom_url = st.text_input(
            "Image Web URL:",
            value="https://thumb.wikimedia.org/wikipedia/commons/thumb/6/64/Monstera_deliciosa_DSC02605.jpg/960px-Monstera_deliciosa_DSC02605.jpg",
            placeholder="https://.../plant_image.jpg",
            label_visibility="collapsed"
        )
    with url_input_col2:
        url_fetch_btn = st.button("📥 Load URL", type="primary", use_container_width=True)

    if url_fetch_btn:
        with st.spinner("Downloading image from URL..."):
            img, err = fetch_image_from_url(custom_url)
            if img is not None:
                st.session_state.current_img_rgb = img
                st.session_state.current_img_title = "Direct URL Web Image"
                st.session_state.selected_img_id = "url_direct"
                st.rerun()
            else:
                st.error(err)

# --- 3. PRESET TEST SAMPLE MODE ---
elif input_source == "📁 Preset Test Sample":
    sample_choice = st.selectbox(
        "Choose a test photo:",
        [
            "Dried Monstera Deliciosa Wilted Leaf (Dried_Monstera_Deliciosa_Leaf.jpg)",
            "Monstera deliciosa Indoor Ceiling & Light (Monstera_deliciosa_DSC02605.jpg)",
            "Monstera deliciosa Clustered Garden #2 (Monstera_deliciosa_22zz.jpg)",
            "Monstera deliciosa Clustered Garden #1 (Monstera_deliciosa_21zz.jpg)",
            "Dieffenbachia with Background Plants (dumbane-dieffenbachia.jpg)",
            "Pothos in Green-Striped Pot (plantimage.jpg)",
            "Real Potted Plant with Visible Pot & Soil",
            "Real Tomato Leaf with Early Blight (Alternaria solani)",
            "Real Healthy Green Leaf with Clear Margins & Veins"
        ]
    )
    if "Dried_Monstera" in sample_choice:
        sample_path = "Dried_Monstera_Deliciosa_Leaf.jpg"
    elif "DSC02605" in sample_choice:
        sample_path = "Monstera_deliciosa_DSC02605.jpg"
    elif "Monstera_deliciosa_22zz" in sample_choice:
        sample_path = "Monstera_deliciosa_22zz.jpg"
    elif "Monstera_deliciosa_21zz" in sample_choice:
        sample_path = "Monstera_deliciosa_21zz.jpg"
    elif "Dieffenbachia" in sample_choice:
        sample_path = "dumbane-dieffenbachia.jpg"
    elif "Striped" in sample_choice:
        sample_path = "plantimage.jpg"
    elif "Pot" in sample_choice:
        sample_path = "samples/potted_plant_sample.jpg"
    elif "Blight" in sample_choice:
        sample_path = "samples/real_diseased_leaf.jpg"
    else:
        sample_path = "samples/real_healthy_leaf.jpg"

    if sample_path and os.path.exists(sample_path):
        if st.session_state.last_preset_choice != sample_choice:
            pil_img = Image.open(sample_path).convert("RGB")
            st.session_state.current_img_rgb = np.array(pil_img)
            st.session_state.current_img_title = sample_choice
            st.session_state.selected_img_id = f"preset_{sample_choice}"
            st.session_state.last_preset_choice = sample_choice

# --- 4. UPLOAD YOUR OWN IMAGE MODE ---
else:
    uploaded_file = st.file_uploader(
        "Upload a plant image (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.session_state.current_img_rgb = np.array(pil_img)
        st.session_state.current_img_title = uploaded_file.name
        st.session_state.selected_img_id = f"upload_{uploaded_file.name}"

# -------------------------------------------------------------
# CHECK IF IMAGE IS LOADED
# -------------------------------------------------------------
current_img_rgb = st.session_state.current_img_rgb

if current_img_rgb is None:
    st.info("👈 Please select or search for an image above to begin preprocessing.")
    st.stop()

# Display input summary
h, w = current_img_rgb.shape[:2]
img_bgr = cv2.cvtColor(current_img_rgb, cv2.COLOR_RGB2BGR)
is_sharp, blur_score = check_blur(img_bgr)

# Top Bar with original image details
with st.expander(f"📷 Active Input: {st.session_state.current_img_title} ({w} × {h} px)", expanded=False):
    col_raw1, col_raw2 = st.columns([1, 2])
    with col_raw1:
        st.image(current_img_rgb, caption="Raw Input Image", use_container_width=True)
    with col_raw2:
        st.write(f"🏷️ **Title:** `{st.session_state.current_img_title}`")
        st.write(f"📏 **Dimensions:** `{w} × {h}` px")
        st.write(f"🔍 **Sharpness (Laplacian Variance):** `{blur_score:.1f}` ({'Sharp' if is_sharp else 'Blurry'})")
        st.write(f"🧠 **Segmentation Model:** `{model_name}`")
        st.write(f"📐 **Safety Margin Padding:** `{padding_val}%`")
        if filter_pot_soil:
            st.success("🛡️ **Pot & Soil Filter:** Active (non-plant pots and soil will be removed).")
        else:
            st.warning("⚠️ **Pot & Soil Filter:** Disabled.")

# -------------------------------------------------------------
# RUN PREPROCESSING PIPELINES
# -------------------------------------------------------------
st.markdown("---")
with st.spinner(f"🔄 Processing '{st.session_state.current_img_title}' through Method 1 (Species ID) & Method 2 (Disease ID)..."):
    res_species = preprocessor.process_for_species_identification(
        current_img_rgb, 
        bg_color=bg_rgb,
        model_name=model_name,
        filter_pot_soil=filter_pot_soil,
        preserve_leaf_shape=preserve_leaf_shape,
        padding_ratio=padding_ratio,
        foliage_scope=foliage_scope
    )
    res_disease = preprocessor.process_for_disease_identification(
        current_img_rgb, 
        bg_color=bg_rgb,
        model_name=model_name,
        filter_pot_soil=filter_pot_soil,
        preserve_leaf_shape=preserve_leaf_shape,
        padding_ratio=max(0.06, padding_ratio - 0.04),
        foliage_scope=foliage_scope
    )

# -------------------------------------------------------------
# CROPPED PLANT RESULTS: METHOD 1 vs METHOD 2
# -------------------------------------------------------------
st.markdown("## 🎯 Pure Plant Cropped Outputs")
st.caption(f"Showing results for: **{st.session_state.current_img_title}**. Leaf shapes & margins preserved. Background clutter, pots, and soil removed.")

col1, col2 = st.columns(2)

# --- METHOD 1 COLUMN ---
with col1:
    st.markdown("""
    <div class="method-card">
        <h3 style="color: #1b5e20; margin: 0 0 8px 0;">Method 1: Plant Species Identification</h3>
        <p style="margin: 0; color: #37474f; font-size: 0.95rem;">
            <strong>Core Focus:</strong> Structural architecture, margin serration sharpness, petiole, and aspect-ratio preservation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    img_species = res_species["cropped_image"]
    sh, sw = img_species.shape[:2]
    st.image(img_species, caption=f"Cropped Plant for Species ID ({sw} × {sh} px)", use_container_width=True)

    # Download button for Method 1
    pil_species = Image.fromarray(img_species)
    buf_species = io.BytesIO()
    pil_species.save(buf_species, format="PNG")
    st.download_button(
        label="⬇️ Download Cropped Plant (Species ID)",
        data=buf_species.getvalue(),
        file_name="cropped_plant_species_id.png",
        mime="image/png",
        use_container_width=True
    )

    with st.expander("📋 Preprocessing Steps Applied (Method 1)", expanded=True):
        for step in res_species["pipeline_steps"]:
            st.markdown(f"- {step}")

# --- METHOD 2 COLUMN ---
with col2:
    st.markdown("""
    <div class="method-card-disease">
        <h3 style="color: #e65100; margin: 0 0 8px 0;">Method 2: Disease & Health Identification</h3>
        <p style="margin: 0; color: #37474f; font-size: 0.95rem;">
            <strong>Core Focus:</strong> Necrotic lesion integrity, zero-hole lamina protection, spore texture enhancement, and genuine hue retention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    img_disease = res_disease["cropped_image"]
    dh, dw = img_disease.shape[:2]
    st.image(img_disease, caption=f"Cropped Plant for Disease ID ({dw} × {dh} px)", use_container_width=True)

    # Download button for Method 2
    pil_disease = Image.fromarray(img_disease)
    buf_disease = io.BytesIO()
    pil_disease.save(buf_disease, format="PNG")
    st.download_button(
        label="⬇️ Download Cropped Plant (Disease ID)",
        data=buf_disease.getvalue(),
        file_name="cropped_plant_disease_id.png",
        mime="image/png",
        use_container_width=True
    )

    with st.expander("📋 Preprocessing Steps Applied (Method 2)", expanded=True):
        for step in res_disease["pipeline_steps"]:
            st.markdown(f"- {step}")

# -------------------------------------------------------------
# SIDE-BY-SIDE COMPARISON & WHY THEY DIFFER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Side-by-Side Comparison: Raw vs Method 1 vs Method 2")

comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 1])

with comp_col1:
    st.image(current_img_rgb, caption="Raw Input (with Clutter, Pot, or Background)", use_container_width=True)

with comp_col2:
    st.image(img_species, caption="Method 1 (Species): Square Letterboxed & Color-Normalized", use_container_width=True)

with comp_col3:
    st.image(img_disease, caption="Method 2 (Disease): Symptomatic Crop with Lamina Integrity", use_container_width=True)

st.markdown("""
| Preprocessing Characteristic | Method 1: For Species Identification | Method 2: For Disease Identification |
| :--- | :--- | :--- |
| **Leaf Shape & Margin Retention** | **Full Margin & Apex Preservation**: Outer boundary envelope captures 100% of leaf tip apexes, margins, and lateral lobes. | **Lamina & Spore Integrity**: Encloses the complete leaf surface so lesions on the perimeter are never truncated. |
| **Safety Margin Padding** | **Square Canvas with Generous Margin**: Safe buffer padding around the plant prevents clipping outward-reaching leaves. | **Symptomatic Crop with Margin**: Tight framing with safe margin buffer around foliage edges. |
| **Complex Background Handling** | **Dichotomous Neural Matting (BiRefNet)**: Separates foreground target plant even when the background is filled with other plants. | **Dichotomous Neural Matting (BiRefNet)**: Disentangles foreground leaves from background foliage clutter. |
| **Pot & Soil Rejection** | **Filtered Out**: Strips pots and soil beds so species classifier doesn't memorize pot shapes or container colors. | **Filtered Out**: Eliminates soil texture and terracotta hues so models don't confuse dirt with necrotic root/stem rot. |
| **Color Treatment** | **Gray-World White Balancing**: Removes sunlight/shadow casts so leaf color is standardized across environments. | **Genuine Color Retention**: Keeps natural lesion hues intact (yellow chlorosis halos, brown necrotic dead tissue). |
| **Lamina & Lesion Protection** | Focuses on crisp leaf boundaries and serrations across all leaves. | **Hole-Free Solid Lamina**: Encloses 100% of internal lesion spots, ensuring dead/brown tissue is never cut out. |
""")
