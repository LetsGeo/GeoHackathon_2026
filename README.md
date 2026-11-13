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

### The app's pages

| Page | What's there |
|---|---|
| **Overview** | Summary cards and a status table: surveys, shapes, horizons, wells, analysis |
| **Seismic viewer** | Baseline and monitor side by side, the morph between them, and the slice toolbar with the **Layers** panel (shapes, horizons, Gemma horizons) and **Display** panel (colour map, range, resolution) |
| **Analysis** | Whole-volume and current-slice change measurements, a summary in English plus Malay / Chinese / Tamil, PDF export |
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
| `polygon.py` | Hexagon geometry: one hexagon, or a centre hexagon with rings around it (2 rings = 19 hexagons, about 250 m across each). Used for the hexagon grid on time slices and by the hexagon scripts below. |
| `wells.py` | The **Wells** page: adding wells (LAS file or table), the well map and sections with log tracks, and the well list. |
| `welllogs.py` | Well-log processing behind the Wells page: reads LAS / checkshot / deviation files, converts depth to two-way time, and finds reservoir candidates (Gemma picks the best one, or a rule engine if Gemma is unreachable). |
| `ai.py` | The single place the app talks to Gemma: server address and model, the reachability check, translations of the analysis summary, and `suggest_horizons()` for the Gemma horizons card. |

### Stand-alone analysis scripts

These aren't loaded by the app. Their default paths point at the hackathon server
(`/data/epic20/APGCE/...`), so change those settings at the top of each file to run them
elsewhere.

| File | What it does |
|---|---|
| `well_plot.py` | Well-log reservoir finder for one LAS file: plots Gamma Ray, Neutron, Density and Resistivity, flags candidate intervals with simple rules, then asks Gemma to pick the reservoir's top and base and explain why. |
| `polygon_ilcl_pics.py` | Loads and aligns the baseline and monitor volumes, builds the 19-hexagon polygon, and saves an inline and a crossline picture through each hexagon, named `hex<ID>_IL_<inline>.png` / `hex<ID>_CL_<crossline>.png`. |
| `gemma_ilcl.py` | Sends those hexagon pictures to Gemma, which scores each one 0–100 % for how likely it crosses a reservoir. Writes `scores.csv` (one score per hexagon), `pictures.csv` and `results.json` into a `gemma/` folder next to the pictures. |
| `sealing_faults.py` | Ranks hexagons around a well by how likely they hold a sealing fault, using the hexagon table, the well and the horizons (dip, roughness, offsets between neighbours). Gemma gives the verdict and reason; a rule engine is the fallback. Works from the command line or as functions. |

### Data and other files

| Path | What's in it |
|---|---|
| `assets/Logo.png` | The app logo in the sidebar. |
| `assets/Gemma.png` | The Gemma icon on the Gemma horizons card. |
| `horizons/TOP-UTSIRA-DEMO.txt`, `horizons/BASE-UTSIRA-DEMO.txt` | **Synthetic** demo horizons for the Sleipner survey, for trying the Horizons layer. Not real interpretation. |
| `.gitignore` | Keeps virtual environments, pictures (`*.png`, `*.jpg`), NumPy caches (`*.npy`, `*.npz`) and PDFs out of the repo. |

### Not in the repo (Kept Locally)

| Path | What it is |
|---|---|
| `data/` | The baseline and monitor SEG-Y files (hundreds of MB each). Put your surveys here, or anywhere, and point **Data sources** at them. |
| `cache/` | Created automatically on the first load of each survey. Safe to delete. |
| `wells/` | Synthetic demo well (`SLEIPNER-DEMO-1.las` with its checkshot and deviation files) for trying the Wells page. |
| `hexagons/` | Synthetic demo hexagon table (`TEST-HEXAGONS-DEMO.txt`) for `sealing_faults.py`. |

---

## How the app's files fit together

```text
app.py ─┬─ styles.py        look and feel (imported by every UI file)
        ├─ loading.py       loading overlay
        ├─ shapefiles.py    Shapes layer: read / write shapefiles
        ├─ horizons.py      Horizons layer
        ├─ polygon.py       hexagon grid on time slices
        ├─ wells.py ─── welllogs.py ─── ai.py
        └─ ai.py            Gemma: translations, horizon suggestions
```
