library(ebirdst)

args <- commandArgs(trailingOnly = TRUE)
species_list <- readLines(args[1])
output_path <- args[2]

valid_species <- ebirdst_runs[1]

for (species in species_list) {
  if (species == "") break

  if (!any(valid_species == species)) {
    cat("Invalid input, could not download range:", species, "\n")
    next
  }

  ebirdst_download(
    species = species,
    path = output_path,
    pattern = "range_smooth_mr",
    force = TRUE
  )
}
