# import argparse
# import os
# from repopulation import analyze_tile_torch, analyze_geotiffs
# from util import read_transmission_csv, createdir_for_file

# def main(args):
#     # Reads and transltes the resistance dictionary.
#     transmission_d = read_transmission_csv(args.resistance)
#     # Creates output folders, if missing.
#     createdir_for_file(args.repopulation)
#     do_gradient = args.gradient is not None

#     # Do the bird run
#     # builds the analysis function, which is passed to analyze_geotiffs below.
#     analysis_fn = analyze_tile_torch(
#         num_simulations=args.num_simulations,
#         hop_length=args.hop_distance,
#         total_spreads=args.num_spreads,
#         seed_density=args.seed_density,
#         produce_gradient=do_gradient,
#     )
#     # Performs the analysis
#     analyze_geotiffs(
#         args.habitat, args.terrain, transmission_d,
#         analysis_fn=analysis_fn,
#         single_tile=True,
#         generate_gradient=do_gradient,
#         output_repop_fn=args.repopulation,
#         output_grad_fn=args.gradient,
#     )

# if __name__ == '__main__':
#     # main(argparse.Namespace(habitat='/tests/assets/habitat_uint8.tif', terrain='/tests/assets/terrain_uint8.tif', resistance='tests/assets/transmission_refined_0.5.csv', repopulation='./repopulation_uint8.tif', gradient=None, hop_distance=4, num_spreads=400, num_simulations=2, seed_density=4))
#     # main(argparse.Namespace(habitat='/Users/nvalett/Documents/Natalie/Species Dist Research/Code/Connectivity-Package/ecoscape/tests/assets/habitat_float32.tif', terrain='/Users/nvalett/Documents/Natalie/Species Dist Research/Code/Connectivity-Package/ecoscape/tests/assets/terrain_float32.tif', resistance='/Users/nvalett/Documents/Natalie/Species Dist Research/Code/Connectivity-Package/ecoscape/tests/assets/transmission_refined_0.5.csv', repopulation='/Users/nvalett/Documents/Natalie/Species Dist Research/Code/Connectivity-Package/ecoscape/tests/assets/repopulation_float32.tif', gradient=None, hop_distance=4, num_spreads=15, num_simulations=50, batch_size=1, seed_density=4))
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-H', '--habitat', type=os.path.abspath, default=None,
#                         help='Filename to a geotiff of the bird\'s habitat.')
#     parser.add_argument('-T', '--terrain', type=os.path.abspath, default=None,
#                         help='Filename to a geotiff of the terrain.')
#     parser.add_argument('-R', '--resistance', type=os.path.abspath, default=None,
#                         help='Filename to a CSV dictionary of the terrain value resistance.')
#     parser.add_argument('-r', '--repopulation', type=os.path.abspath, default=None,
#                         help='Filename to output geotiff file for repopulation.')
#     parser.add_argument('-g', '--gradient', type=os.path.abspath, default=None,
#                         help='Filename to output geotiff file for gradient.')
    
#     parser.add_argument('-d', '--hop_distance', type=int, default=4,
#                         help='Distance the bird can travel in one flight.')
#     parser.add_argument('-s', '--num_spreads', type=int, default=15,
#                         help='Number of spreads for the model to compute.')
#     parser.add_argument('-b', '--batch_size', type=int, default=1,
#                         help='Batch size of each simulation. 1 if running on one core, otherwise 400 is best for GPU')
#     parser.add_argument('-S', '--num_simulations', type=int, default=200,
#                         help='Number of repopulation simulations to be run.')
#     parser.add_argument('-D', '--seed_density', type=int, default=4,
#                         help='Density of random seeds in the simulation.')

#     args = parser.parse_args()
#     main(args)
    
