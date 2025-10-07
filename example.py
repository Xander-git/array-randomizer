# Example usage and demonstration
if __name__ == "__main__":
    # Configuration
    config = PlateConfig(
        n_strains=180,
        n_rows=8,
        n_columns=12,
        n_negatives=6,
        n_replicates=3,
        random_seed=42  # Set seed for reproducible results
    )

    print("=" * 60)
    print("PLATE LAYOUT GENERATION SYSTEM")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Strains: {config.n_strains}")
    print(f"  Plate format: {config.n_rows} x {config.n_columns} ({config.n_rows * config.n_columns} wells)")
    print(f"  Negatives per plate: {config.n_negatives}")
    print(f"  Replicates: {config.n_replicates}")
    print(f"  Random seed: {config.random_seed if config.random_seed is not None else 'None (random)'}")

    # Generate layouts
    print("\n" + "-" * 60)
    print("STEP 1: Creating base plate set...")
    print("-" * 60)

    generator = PlateLayoutGenerator(config)
    base_set = generator.create_base_set()

    print(f"✓ Generated {generator.n_plates} plates")
    print(f"✓ Usable wells per plate: {generator.usable_wells}")

    # Generate replicates
    print("\n" + "-" * 60)
    print("STEP 2: Generating replicates with edge alternation...")
    print("-" * 60)

    results = generator.generate_replicates()

    print(f"✓ Generated {len(results['replicates'])} replicates")
    if results['metadata']['alternation_scores']:
        avg_score = np.mean(results['metadata']['alternation_scores'])
        print(f"✓ Average alternation score: {avg_score:.2%}")

    # Create visualizer
    print("\n" + "-" * 60)
    print("STEP 3: Creating visualizations...")
    print("-" * 60)

    viz = PlateVisualizer(generator)

    # Example visualizations
    print("\nGenerating example visualizations:")
    print("  1. Plate layout for Replicate 1, Plate 1")
    fig1, _ = viz.visualize_plate_layout(0, 0)
    plt.savefig('plate_layout_rep1_plate1.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: plate_layout_rep1_plate1.png")

    if config.n_replicates > 1:
        print("  2. Edge alternation comparison (Rep 1 vs Rep 2)")
        fig2 = viz.visualize_edge_alternation(0, 1, 0)
        plt.savefig('edge_alternation_comparison.png', dpi=150, bbox_inches='tight')
        print("     ✓ Saved: edge_alternation_comparison.png")

    print("  3. Negative distribution heatmap")
    fig3 = viz.visualize_negative_distribution()
    plt.savefig('negative_distribution.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: negative_distribution.png")

    print("  4. Statistical summary dashboard")
    fig4 = viz.visualize_summary()
    plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: summary_dashboard.png")

    # Example: Track specific strain
    print("  5. Strain S0042 across all replicates")
    fig5 = viz.visualize_all_replicates('S0042')
    plt.savefig('strain_S0042_replicates.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: strain_S0042_replicates.png")

    # Verification checklist
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKLIST")
    print("=" * 60)

    # Check negative uniqueness
    all_unique = True
    for plate_id in range(generator.n_plates):
        positions_per_rep = []
        for rep_id in range(config.n_replicates):
            pos_set = set(results['metadata']['negative_positions'][rep_id][plate_id])
            positions_per_rep.append(pos_set)

        for i in range(len(positions_per_rep) - 1):
            if positions_per_rep[i] == positions_per_rep[i + 1]:
                all_unique = False
                break

    print(f"✓ Negative uniqueness: {'PASS' if all_unique else 'FAIL'}")

    # Check strain consistency
    strain_consistent = True
    for rep_id in range(config.n_replicates):
        for plate_id in range(generator.n_plates):
            strains_base = set()
            strains_rep = set()

            for row in range(config.n_rows):
                for col in range(config.n_columns):
                    cell_base = base_set[plate_id, row, col]
                    cell_rep = results['replicates'][rep_id][plate_id, row, col]

                    if cell_base is not None and cell_base != "negative":
                        strains_base.add(cell_base)
                    if cell_rep is not None and cell_rep != "negative":
                        strains_rep.add(cell_rep)

            if strains_base != strains_rep:
                strain_consistent = False
                break

    print(f"✓ Strain consistency: {'PASS' if strain_consistent else 'FAIL'}")

    # Check edge alternation
    if results['metadata']['alternation_scores']:
        avg_score = np.mean(results['metadata']['alternation_scores'])
        alternation_pass = avg_score > 0.3  # At least 30% alternation
        print(f"✓ Edge alternation (avg {avg_score:.2%}): {'PASS' if alternation_pass else 'FAIL'}")

    # Check randomization
    if config.n_replicates > 1:
        identical = True
        for plate_id in range(generator.n_plates):
            if not np.array_equal(results['replicates'][0][plate_id],
                                 results['replicates'][1][plate_id]):
                identical = False
                break
        randomization_pass = not identical
        print(f"✓ Randomization: {'PASS' if randomization_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print("\nAll visualizations saved. Review the images to validate the layout.")
    print(f"\nRandom seed used: {config.random_seed if config.random_seed is not None else 'None (random)'}")
    print("  - Use the same seed to reproduce identical layouts")
    print("  - Set random_seed=None for different layouts each run")
    print("\nTo use this system:")
    print("  1. Modify PlateConfig parameters as needed")
    print("  2. Run generator.create_base_set()")
    print("  3. Run generator.generate_replicates()")
    print("  4. Use PlateVisualizer methods for validation")
    print("\nFor custom analysis, access:")
    print("  - generator.replicates[rep_id][plate_id] for plate arrays")
    print("  - generator.metadata for scores and negative positions")
    print("  - generator.metadata['random_seed'] for the seed used")

    plt.show()
