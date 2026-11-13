# Drill Baby Drill — Reservoir Time Machine

A 4D (time-lapse) seismic workspace built with [Panel](https://panel.holoviz.org/) and
[Bokeh](https://bokeh.org/). Load a **baseline** and a **monitor** SEG-Y survey, compare them
slice by slice, morph from one to the other, measure what changed, and add interpretation on
top: shapes, horizons, wells and a hexagon grid. Gemma (the hackathon's language model) helps
with well picks, horizons, translations and fault screening.

The repo has two kinds of Python files:

- **The app** — everything `panel serve app.py` loads.
- **Stand-alone analysis scripts** — run on their own (terminal or Jupyter) on the hackathon
  server, to produce pictures, scores and tables.

---

## Running the app

```bash
pip install panel==1.4.5 bokeh==3.4.3 numpy pandas segyio matplotlib openai
panel serve app.py --show
```

Then open **Data sources** in the sidebar, point it at the two SEG-Y files and press **Run**.
You can also pass the files when starting the app:

```bash
panel serve app.py --show --args --baseline data/94p10ful --monitor data/10p10ful --baseline-label "1994 survey" --monitor-label "2010 survey"
```

- Tested with Python 3.9, Panel 1.4.5 and Bokeh 3.4.3.
- `matplotlib` is only needed for the PDF export, and `openai` only for Gemma.
- The first time a survey is loaded it is read once and saved to `cache/` (about the size of
  the SEG-Y), so slices are instant afterwards. Delete `cache/` any time to free the space.
- Gemma is only reachable on the hackathon network. Off it, every Gemma feature falls back or
  says it can't connect; the rest of the app works normally.

### The pipeline

1. **Data sources**: load the baseline and monitor surveys (**Run**).
2. **Wells**: load a well; Gemma (or the rule engine) picks its reservoir's top and base.
3. **Seismic viewer**: the 19-hexagon grid sits on the chosen well on time slices; horizons and shapes go on the Layers panel.
4. **Analysis**: **Run analysis** measures the change; **Run hexagon analysis** puts the 19 hexagons round the well at its reservoir top and, in the reservoir window (the well's pick, plus a margin):
   - **3D polygons**: per hexagon, RMS(base) − RMS(mon), the fault attribute → sealing fault probability % and trace % (the hexagon table), and `sealing_faults.py`'s verdict with the well and horizons;
   - **Opportunity score**: each hexagon's central inline and crossline (baseline beside the 4D difference) go to Gemma, which scores reservoir, change and opportunity 0–100 %;
   - **Hydrocarbon volume**: area × thickness × N/G × porosity (from the well's picked interval) × (1 − Sw) ÷ Bo per hexagon, in place and risked by the opportunity score.

   Gemma writes the three sections in English and Malay (written from the numbers when Gemma can't be reached); they appear in the summary, its translations and the PDF, with a map and table page. The card also downloads the hexagon table (`sealing_faults.py`'s input format) and the details.

### The app's pages

| Page | What's there |
|---|---|
| **Overview** | Summary cards and a status table: surveys, shapes, horizons, wells, analysis |
| **Seismic viewer** | Baseline and monitor side by side, the morph between them, and the slice toolbar with the **Layers** panel (shapes, horizons, Gemma horizons) and **Display** panel (colour map, range, resolution) |
| **Analysis** | Whole-volume and current-slice change measurements; the **Hexagon analysis** (3D polygons round the well, opportunity score, hydrocarbon volume) that fills the report's three AI sections; a summary in English plus Malay / Chinese / Tamil; PDF export |
| **Wells** | Load wells from LAS files, see them over the seismic with their log tracks |
| **Data sources** | The two SEG-Y files and the Run button |

---

## Where everything is

### The app

| File | What it does |
|---|---|
| `app.py` | **The app itself.** Loading and caching the SEG-Y surveys, slicing, the plots and the morph, the sidebar pages, the shapes (shapefile) tools, the analysis and its PDF export, and the language and Gemma horizon features. Start here. |
| `styles.py` | **All the design.** Light / dark theme colours, fonts, every widget's CSS, icons, seismic colour maps, plot colours, the loading-screen styling and the oil-splatter easter egg. Change the look here without touching the logic. |
| `loading.py` | The full-screen pumpjack loading overlay shown while **Run** is loading the surveys. |
| `shapefiles.py` | Reads and writes ESRI shapefiles (`.shp` / `.shx` / `.dbf`) — points, lines and polygons — for the Shapes layer. No external GIS library needed. |
| `horizons.py` | The Horizons layer: loads horizon files (Petrel / IESX-style text tables, with or without a header), lets you draw horizons on a section, draws them on the plots and exports them as CSV. |
| `polygon.py` | Hexagon geometry: one hexagon, or a centre hexagon with rings around it (2 rings = 19 hexagons). `SCALE` = 110 lines radius, the same as the hexagon scripts. Used for the hexagon grid on time slices, the hexagon analysis and the scripts below. |
| `wells.py` | The **Wells** page: adding wells (LAS file or table), the well map and sections with log tracks, and the well list. |
| `welllogs.py` | Well-log processing behind the Wells page: reads LAS / checkshot / deviation files, converts depth to two-way time, and finds reservoir candidates (Gemma picks the best one, or a rule engine if Gemma is unreachable). |
| `ai.py` | The single place the app talks to Gemma: server address and model, the reachability check, text and picture requests (`gemma_chat`, `gemma_chat_image`), translations of the analysis summary, and `suggest_horizons()` for the Gemma horizons card. |
| `hexagon_analysis.py` | The hexagon analysis engine behind the Analysis page's **Hexagon analysis** card: the window from the well's reservoir pick, the hexagons round the well, the trace attributes (via `hex_table.py`), `sealing_faults.py`'s verdicts, Gemma's opportunity scores (via `gemma_ilcl.py`'s prompt), the volumes and the three report sections. |
| `hexagon_ui.py` | The **Hexagon analysis** card: Run with its settings and progress, the summary tiles, the map, the table, the best hexagon's sections, the CSV downloads and the PDF page. |

### Stand-alone analysis scripts

These aren't loaded by the app. Their default paths point at the hackathon server
(`/data/epic20/APGCE/...`), so change those settings at the top of each file to run them
elsewhere.

| File | What it does |
|---|---|
| `well_plot.py` | Well-log reservoir finder for one LAS file: plots Gamma Ray, Neutron, Density and Resistivity, flags candidate intervals with simple rules, then asks Gemma to pick the reservoir's top and base and explain why. |
| `polygon_ilcl_pics.py` | Loads and aligns the baseline and monitor volumes (`volume3d.py`), builds the 19-hexagon polygon, and saves inline and crossline pictures through each hexagon, named `hex<ID>_IL_<inline>.png` / `hex<ID>_CL_<crossline>.png`, each with its 4D difference `..._diff.png`. |
| `gemma_ilcl.py` | Sends those hexagon pictures to Gemma: each section's baseline beside its 4D difference (`_diff` picture), scored 0–100 % for reservoir, change and opportunity (reservoir only without a difference picture). Writes `scores.csv` (one row per hexagon), `pictures.csv` and `results.json` into a `gemma/` folder next to the pictures. |
| `volume3d.py` | The 3D SEG-Y loader and baseline / monitor aligner the scripts share (`load_cache_or_load_and_align_3d_pair`), with a cache per pair of input files. |
| `hex_rms.py` | RMS(base) − RMS(mon) for every hexagon and time window, with a map and a profile. |
| `hex_table.py` | The final hexagon table from trace attributes: hexagon, centre IL, centre CL, sealing fault probability %, trace % (`sealing_faults.py`'s input), plus a details file. |
| `sealing_faults.py` | Ranks hexagons around a well by how likely they hold a sealing fault, using the hexagon table, the well and the horizons (dip, roughness, offsets between neighbours). Gemma gives the verdict and reason; a rule engine is the fallback. Works from the command line or as functions. |

### Data and other files

| Path | What's in it |
|---|---|
| `assets/Logo.png` | The app logo in the sidebar. |
| `assets/Gemma.png` | The Gemma icon on the Gemma horizons card. |
| `.gitignore` | Keeps virtual environments, pictures (`*.png`, `*.jpg`), NumPy caches (`*.npy`, `*.npz`) and PDFs out of the repo. |

---

## How the app's files fit together

```text
app.py ─┬─ styles.py        look and feel (imported by every UI file)
        ├─ loading.py       loading overlay
        ├─ shapefiles.py    Shapes layer: read / write shapefiles
        ├─ horizons.py      Horizons layer
        ├─ polygon.py       hexagon grid on time slices
        ├─ wells.py ─── welllogs.py ─── ai.py
        ├─ hexagon_ui.py ── hexagon_analysis.py ─┬─ hex_table.py / hex_rms.py   trace attributes
        │                                        ├─ sealing_faults.py           verdicts (well + horizons)
        │                                        ├─ gemma_ilcl.py               opportunity prompt
        │                                        └─ volume3d.py, polygon.py, ai.py
        └─ ai.py            Gemma: translations, horizon suggestions, pictures
```
