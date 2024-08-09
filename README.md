# Paper on Network Cleaning

Paper on network cleaning investigations for T2E project. Based on EU high density clusters and overturemaps data (which is largely based on OpenStreetMap).

## Installation

Clone this repository to a local working folder.

The PDM package manager is recommended and can be installed on mac per `brew install pdm`.

Packages can then be installed into a virtual environment per `pdm install`.

If using an IDE the `.venv` should be detected automatically by IDEs such as vscode.

## Data

### Madrid Boundary

- Taken from the [madrid-ua-dataset](https://github.com/songololo/madrid-ua-dataset) reference dataset.

### Madrid Street Network

- Taken from the [madrid-ua-dataset](https://github.com/songololo/madrid-ua-dataset) reference dataset.

### Overture Trunk Roads for the EU

```sql
WITH bounds AS (
    SELECT geom
    FROM eu.bounds
)
SELECT
    ROW_NUMBER() OVER () AS uid,
    ne.*
FROM
    overture.network_edges_raw ne
JOIN
    bounds b ON ST_Intersects(b.geom, ne.geom)
WHERE
    ne.highways = '[''trunk'']';
```
