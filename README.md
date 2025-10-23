# Phase Detection using DTW

A simple Python script that finds seismic phase arrivals in waveform data using Dynamic Time Warping (DTW). Basically, you give it a waveform and a template, and it tells you where the best match is.

## What does it do?

The script slides a template waveform across your data and calculates how similar they are at each position using DTW. The spot with the lowest distance = best match = your phase arrival time!

## Quick Start

1. **Install dependencies:**
   ```bash
   conda env create -f environment.yml
   conda activate data
   ```

2. **Edit the config section** at the top of `perform_phase_detection_general.py`:
   ```python
   WAVEFORM_PATH: str = "example_traces/STN_syn.z"  # your data file
   TEMPLATE_PATH: str | None = None                  # or path to template file
   BEG: int = 100                                     # template start (if using data slice)
   TER: int = 120                                     # template end (if using data slice)
   ```

3. **Run it:**
   ```bash
   python perform_phase_detection_general.py
   ```

## How to use your own data

### Option 1: Use part of your waveform as template
Set `TEMPLATE_PATH = None` and define `BEG` and `TER` to extract a template from your waveform:
```python
WAVEFORM_PATH = "path/to/your/data.sac"
TEMPLATE_PATH = None
BEG = 100   # sample index where your template starts
TER = 150   # sample index where your template ends
```

### Option 2: Use a separate template file
Point to a template file (can be any format ObsPy can read):
```python
WAVEFORM_PATH = "path/to/your/data.sac"
TEMPLATE_PATH = "path/to/your/template.sac"
```

## Settings you might want to tweak

### Preprocessing (applied to both waveform and template)
```python
APPLY_DEMEAN = True          # remove mean
APPLY_DETREND = True         # remove linear trend
TAPER_FRACTION = 0.05        # taper edges (5%)
```

### DTW options
```python
DTW_USE_PRUNING = True            # speeds things up
DTW_WINDOW_RATIO = 0.1            # constrains warping (0.1 = 10% of template length)
                                  # set to None for no constraint (slower but more flexible)
DTW_PSI = 0                       # slope constraint (0 = none)
```

### Output options
```python
MAKE_GIF = False                  # set to True if you want a cool animated visualization
                                  # (warning: slower and creates lots of temp files)
SHOW_PROGRESS = True              # show progress bar
SAVE_DIST_CSV = True              # save all DTW distances to distances.csv
PERSIST_GIF_PATH = Path("figures/template_match.gif")  # where to save the GIF
```

## Output

The script prints:
- Best match index (sample number)
- Best match time (absolute time if available)
- DTW distance at best match (lower = better)

If `SAVE_DIST_CSV = True`, you'll also get a `distances.csv` file with all the distance values.

If `MAKE_GIF = True`, you'll get an animated GIF showing the template sliding across your data!

## Tips

- **Fast mode:** Keep `MAKE_GIF = False` for quick runs (especially on long waveforms)
- **DTW window:** Start with `DTW_WINDOW_RATIO = 0.1` - increase if your phases are really stretched/compressed
- **Template length:** Shorter templates = faster but less specific. Longer = slower but more unique.
- **File formats:** Works with anything ObsPy can read (SAC, mseed, etc.)

## Example

Using the included example data:
```bash
python perform_phase_detection_general.py
```

This will search for the template (samples 100-120) in `example_traces/STN_syn.z` and tell you where it best matches.

## Troubleshooting

- **"BEG/TER out of bounds"**: Your template indices are outside your waveform length
- **Slow performance**: Try `MAKE_GIF = False` or increase `DTW_WINDOW_RATIO`
- **No clear minimum**: Your template might not exist in the waveform, or preprocessing might be too aggressive
- **ImportError**: Make sure you activated the conda environment

## What's DTW?

Dynamic Time Warping is a way to measure similarity between two time series that might be slightly stretched or compressed. Unlike simple cross-correlation, DTW can handle timing variations - perfect for seismic phases that don't arrive at exactly the same speed every time!

## References
Kumar, U., Legendre, C. P., Zhao, L., and Chao, B. F., 2022. Dynamic Time Warping as an Alternative to Windowed Cross Correlation in Seismological Applications. Seismological Research Letters. DOI: 10.1785/0220210288
