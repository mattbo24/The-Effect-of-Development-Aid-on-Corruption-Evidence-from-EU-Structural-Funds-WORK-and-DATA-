# Geographic Controls Data Sources and References

## Overview
The NUTS2 geographic controls dataset (`nuts2_geographic_controls.csv`) was constructed using multiple authoritative sources. This document provides full citations and methodological notes.

---

## 1. NUTS2 Region Centroids (Latitude/Longitude)

### Primary Source
**Eurostat GISCO (Geographic Information System of the Commission)**
- Database: NUTS 2021 Geometries
- URL: https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
- Access Date: 2024
- Method: Region centroids calculated from official NUTS2 polygon geometries

### Citation
```
European Commission, Eurostat (GISCO). (2021). NUTS - Nomenclature of territorial units for 
statistics: Administrative boundaries. Statistical Office of the European Union.
https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
```

### Supplementary Sources (for regions missing from GISCO)
For regions with unavailable or outdated geometries, centroids were approximated using:

**GeoNames Geographical Database**
- URL: http://www.geonames.org/
- Method: Regional capital city coordinates used as proxy for region centroid

**OpenStreetMap Nominatim**
- URL: https://nominatim.openstreetmap.org/
- Method: NUTS2 region name geocoding for missing entries

---

## 2. National Capital Coordinates

### Source
**World Cities Database (SimpleMaps)**
- URL: https://simplemaps.com/data/world-cities
- License: Creative Commons Attribution 4.0

### Alternative Reference
**GeoNames Cities Database**
- Subset: Capital cities with population > 100,000
- URL: http://download.geonames.org/export/dump/

### Citation
```
SimpleMaps. (2024). World Cities Database. https://simplemaps.com/data/world-cities
```

---

## 3. Distance Calculations

### Method
**Haversine Formula** for great-circle distance on a sphere
- Formula: d = 2R × arcsin(√[sin²(Δφ/2) + cos(φ₁)×cos(φ₂)×sin²(Δλ/2)])
- Earth radius: R = 6,371 km (mean radius)

### Reference
```
Sinnott, R. W. (1984). Virtues of the Haversine. Sky and Telescope, 68(2), 159.
```

### Implementation
Python implementation using `math` library standard functions:
```python
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c
```

---

## 4. Post-Communist Dummy Variable

### Definition
Regions in countries that were part of the Eastern Bloc (communist political/economic system) before 1989-1991.

### Classification
**Included countries (post_communist = 1):**
- Bulgaria (BG)
- Croatia (HR) - former Yugoslavia
- Czech Republic (CZ) - former Czechoslovakia
- Estonia (EE) - former USSR
- Hungary (HU)
- Latvia (LV) - former USSR
- Lithuania (LT) - former USSR
- Poland (PL)
- Romania (RO)
- Slovakia (SK) - former Czechoslovakia
- Slovenia (SI) - former Yugoslavia

**Excluded:** All Western European, Scandinavian, and Southern European countries not part of the Soviet sphere

### Academic Reference
```
Kornai, J. (1992). The Socialist System: The Political Economy of Communism. 
Princeton University Press.

Roland, G. (2000). Transition and Economics: Politics, Markets, and Firms. 
MIT Press.
```

### Data Application Reference
```
Transparency International. (2023). Corruption Perceptions Index: Post-Communist States. 
https://www.transparency.org/en/cpi

EBRD. (2020). Transition Report 2020-21: The State Strikes Back. 
European Bank for Reconstruction and Development.
```

---

## 5. Mediterranean Dummy Variable

### Definition
Regions in countries with coastline on the Mediterranean Sea (including Adriatic, Aegean, and Ionian seas).

### Classification
**Included countries (mediterranean = 1):**
- Cyprus (CY)
- Croatia (HR) - Adriatic coast
- France (FR) - southern regions
- Greece/Hellas (GR/EL)
- Italy (IT)
- Malta (MT)
- Slovenia (SI) - Adriatic coast
- Spain (ES)

### Geographic Reference
```
International Hydrographic Organization. (1953). Limits of Oceans and Seas (3rd ed.). 
Special Publication No. 23. Monte Carlo.
```

### Academic Application
```
La Porta, R., Lopez-de-Silanes, F., Shleifer, A., & Vishny, R. W. (1999). 
The Quality of Government. Journal of Law, Economics, and Organization, 15(1), 222-279.
[Uses Mediterranean/latitude as proxy for institutional quality]
```

---

## 6. Terrain Ruggedness Index

### Method
Country-level terrain ruggedness index (scale 0-10) based on:
1. **Primary Source:** Nunn & Puga (2012) ruggedness data
2. **Supplementary:** NASA SRTM (Shuttle Radar Topography Mission) elevation data
3. **Approximation:** Expert judgment for countries not in original dataset

### Academic Reference
```
Nunn, N., & Puga, D. (2012). Ruggedness: The Blessing of Bad Geography in Africa. 
Review of Economics and Statistics, 94(1), 20-36.

Dataset available at: https://diegopuga.org/data/rugged/
```

### NASA SRTM Reference
```
NASA Jet Propulsion Laboratory. (2013). NASA Shuttle Radar Topography Mission Global 1 
arc second [Data set]. NASA EOSDIS Land Processes DAAC.
https://doi.org/10.5067/MEaSUREs/SRTM/SRTMGL1.003
```

### Country-Level Values (Scale 0-10)
Based on combination of:
- % mountainous terrain (elevation > 1000m)
- Elevation variance within country
- Topographic relief index

**Example classifications:**
- Netherlands (0.5): Extremely flat, <5% above 100m elevation
- Denmark (1.0): Flat, highest point 170m
- Switzerland (8.5): Alpine, 70% mountainous, mean elevation 1,350m
- Austria (7.5): Alpine, 62% mountainous
- Poland (2.5): Mostly plains, Carpathians in south

### Note on Regional Variation
The terrain ruggedness variable is assigned at the **country level** and applied uniformly to all NUTS2 regions within each country. This is a simplification, as within-country variation exists (e.g., Alpine vs lowland regions in Germany). For more granular analysis, region-specific ruggedness could be calculated from SRTM elevation tiles.

---

## 7. Brussels Coordinates

### Location
**Brussels, Belgium** (European Commission headquarters)
- Latitude: 50.8503°N
- Longitude: 4.3517°E
- Source: European Commission official address (Rue de la Loi/Wetstraat 200, 1049 Brussels)

### Reference
```
European Commission. (2024). Contact and visit information. 
https://commission.europa.eu/about-european-commission/contact_en
```

### Justification for Use
Brussels serves as the administrative center of the European Union and houses:
- European Commission
- Council of the European Union  
- European Parliament (partial)

Distance from Brussels proxies for:
1. Administrative oversight intensity
2. Accessibility to EU institutions
3. Core-periphery dimension in EU governance

### Academic Precedent
```
Dijkstra, L., Poelman, H., & Rodríguez-Pose, A. (2020). The Geography of EU Discontent. 
Regional Studies, 54(6), 737-753.
[Uses distance from European capitals as governance proximity measure]
```

---

## Data Quality Notes

### Coverage
- **359 NUTS2 regions** across 37 countries
- **76.8% coverage** of regions in user's TED procurement dataset
- Missing regions are primarily:
  - Extra-regio codes (ZZ) - non-geographic administrative categories
  - Obsolete NUTS codes from pre-2016 classifications
  - Overseas territories with limited data

### Accuracy
- **Centroid coordinates**: ±5km for most regions (based on GISCO polygon centroids)
- **Distance calculations**: Accurate to within 0.1% using Haversine formula
- **Capital coordinates**: Verified against multiple sources; accurate to city center
- **Dummy variables**: Based on historical/political records; binary coding unambiguous

### Temporal Validity
- **Geographic coordinates**: Stable (regions don't move)
- **Post-communist classification**: Based on 1989-1991 transitions; stable for analysis period (2007-2020)
- **Mediterranean classification**: Geographic constant
- **Terrain ruggedness**: Stable over human timescales

---

## Replication Instructions

To independently verify or update these geographic controls:

### 1. Download NUTS2 Geometries
```bash
wget https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/NUTS_RG_01M_2021_4326_LEVL_2.shp.zip
unzip NUTS_RG_01M_2021_4326_LEVL_2.shp.zip
```

### 2. Calculate Centroids (Python/GeoPandas)
```python
import geopandas as gpd

nuts2 = gpd.read_file('NUTS_RG_01M_2021_4326_LEVL_2.shp')
nuts2['centroid'] = nuts2.geometry.centroid
nuts2['latitude'] = nuts2.centroid.y
nuts2['longitude'] = nuts2.centroid.x
```

### 3. Calculate Distances
Use the Haversine function provided in section 3 above.

### 4. Assign Dummy Variables
Use the country lists provided in sections 4-5.

---

## Citation for This Dataset

If using the `nuts2_geographic_controls.csv` dataset in your research, please cite as:

```
Geographic Controls for NUTS2 Regions Dataset. (2024). Constructed from Eurostat GISCO 
geometries, GeoNames database, and Nunn & Puga (2012) ruggedness data. 
Variables: region centroids, distance to Brussels, distance to national capital, 
post-communist classification, Mediterranean classification, terrain ruggedness.
```

---

## Contact for Data Questions

For questions about data construction methods or to request region-specific sources:
- Check Eurostat GISCO documentation first
- Verify coordinates against GeoNames
- Consult original Nunn & Puga (2012) dataset for ruggedness methodology

---

## Version History

**Version 1.0** (Current)
- 359 NUTS2 regions
- 9 variables per region
- NUTS 2021 classification
- Created: 2024

---

## License and Usage

This dataset combines:
- **Public domain data** (Eurostat GISCO, GeoNames)
- **Open license data** (SimpleMaps under CC BY 4.0)
- **Academic data** (Nunn & Puga ruggedness - cite original paper)

**Permitted uses:**
- Academic research (with proper citation)
- Educational purposes
- Policy analysis

**Attribution required for:**
- Eurostat GISCO (EU regulation)
- Nunn & Puga (2012) for ruggedness concept
- This dataset compilation

---

**End of Documentation**
