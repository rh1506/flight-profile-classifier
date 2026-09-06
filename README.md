# Flight Profile Classifier

Automated classification of drone imagery into 8 flight profiles for cell tower and infrastructure inspection. Transfer learning model trained on 28,000+ real drone photos with 90.3% accuracy.

## 🚀 Try It Live

**[Launch the Flight Profile Classifier](https://rh1506-flight-profile-classifier.streamlit.app)**

Password: `drone123`

Upload your drone photos and they'll be automatically sorted into 8 flight profiles within seconds.

---

## Problem

Infrastructure inspection companies conduct 8 distinct flight profiles per tower to capture different angles and details:

- **Cable Run** — side view following cable routes  
- **Center In** — 360° orbit around tower, centered
- **Center Out** — 360° orbit facing away from tower
- **Compound Flight** — low-altitude equipment survey
- **Downlook** — high-angle descent capturing arrays
- **Overall** — nadir top-down + anchor detail
- **Tower Flight** — multi-orbit vertical coverage
- **Uplook** — ground-level 360° upward sweep

**Current workflow:** Manually sort 500+ drone photos per job into 8 folders. Time-consuming, error-prone, not scalable.

**Solution:** ML classifier that auto-sorts flight profile photos with 90.3% accuracy.

---

## Model

**Architecture:** Transfer Learning with ResNet18
- Pre-trained on ImageNet (14M photos)
- Early layers frozen (image feature extraction already learned)
- Final layer: 8-class classifier trained on real drone imagery
- Input: 224×224 RGB images
- Output: Flight profile class (0-7)

**Why Transfer Learning?**
- Converges in 10 epochs vs. 15+ for from-scratch model
- Learns flight profiles faster with limited data
- Leverages ImageNet's knowledge of shapes, angles, lighting
- Production-ready accuracy without massive dataset

---

## Performance

**Overall Accuracy: 90.3%** on 28,853 real tower inspection images

| Profile | Photos | Accuracy |
|---------|--------|----------|
| Center Out | 888 | 100.0% |
| Uplook | 2,804 | 100.0% |
| Center In | 3,000 | 99.5% |
| Tower Flight | 15,051 | 92.5% |
| Cable Run | 999 | 80.6% |
| Compound Flight | 3,240 | 83.2% |
| Overall | 90 | 68.9% |
| Downlook | 2,781 | 67.9% |

**Generalization:** 90.3% vs overfitting indicates real learning of flight profile patterns, not memorization.

---

## Dataset

**28,853 real drone photographs** from 45 infrastructure inspections:

- DJI drone inspections (Mechanical Vision)
- Multiple tower types: monopole, lattice, guyed
- Varying lighting, weather, gimbal angles (±2°)
- Represents actual working conditions

**Training time:** ~25 minutes (10 epochs, batch size 32)  
**Inference speed:** <100ms per image on CPU

---

## Features

✅ **Automated Sorting** — Classify 600+ photos in seconds  
✅ **Organized Output** — Photos sorted into 8 flight profile folders  
✅ **Download Options** — ZIP download or save directly to computer  
✅ **Password Protected** — Only authorized pilots can access  
✅ **No Installation** — Works entirely in web browser  
✅ **Real Data** — Trained on 28,000+ actual drone photos  

---

## Usage

### Online (No Installation)

1. Visit [Flight Profile Classifier](https://rh1506-flight-profile-classifier.streamlit.app)
2. Enter password: `drone123`
3. Upload JPG/PNG photos
4. Wait for classification (~1-2 sec per photo)
5. Download ZIP or save to computer

### Local Installation

```bash
pip install torch torchvision pillow numpy streamlit
streamlit run app.py
```

---

## Business Impact

**Before:** 2–3 hours manually sorting 500 photos per inspection  
**After:** 5–10 minutes automated sorting with 90.3% accuracy  
**ROI:** ~2 hours saved per job × 10+ jobs/month = 20+ hours saved monthly

Remaining 9.7% manual review < 30 mins/job. Scales with additional training data.

---

## How It Works

1. **Upload** — Drag and drop 100+ drone photos
2. **Classify** — Model predicts flight profile for each photo
3. **Organize** — Photos automatically sorted into folders by flight type
4. **Download** — Get ZIP with organized photo structure

Example output:
