# Analysis provenance and reproducibility

The first public snapshot included frozen intermediate tables alongside analysis code. The current source distribution contains the analysis scripts and signature definitions. The source distribution excludes input matrices, fitted objects and result tables. Contact the corresponding authors about access to study-specific inputs. The [input guide](DATA_SOURCES.md) lists the required inputs, and the [figure map](MANUSCRIPT_CODE_MAP.md) describes the steps covered by the repository.

## Malignant-cell program assignment

In the first snapshot, taking the maximum of the archived six-program usage matrix disagreed with the archived dominant-program labels for 602 of 6,165 cells. Of these discrepancies, 256 changed whether a cell was assigned to P1. The archived usage matrix gave 1,753 P1 assignments, and the archived label table gave 1,857. Those files should not be used interchangeably.

The current [projection script](01_single_cell_selection_and_cnmf/project_programs.py) uses the consensus weights for training cells and nonnegative least squares for other cells, with expression normalized over the projection features. It normalizes each cell's weights to sum to one. The [cohort check](01_single_cell_selection_and_cnmf/check_cell_universe.py) checks cell-set membership and counts for all six programs. Agreement of the counts alone does not establish agreement for each cell. A per-cell audit requires the authoritative label table and the usage matrix generated from the analysis inputs, aligned by cell ID. These inputs are not in the source distribution.

## Figure methods

- The current [regulon analysis](02_redox_branch_and_tf_activity/regulon_activity.py) calls decoupler 2.1.5 ULM directly. The custom TF implementation in the first snapshot is a different implementation and should not be used to regenerate Figure 2F.
- Supplementary Figure 2C displays the 15 highest and 10 lowest module effects from the P1 versus non-P1 ranking of 168 scFEA modules. The upstream flux inference and the ranked module table are separate from this source distribution.

The repository provides code for the steps in the figure map. Full reproduction also requires the stated analysis inputs and the separate upstream workflows.
