#!/usr/bin/env nextflow

// 
// Workflows for vizualization of the reults
// 


workflow visualization {
    take: 
        combined_metrics
        main_outdir
        rt_unit
        output_column_order
        spikein_columns
        output_table_type
        search_spike_ins
        height_barplots
        width_barplots
        height_pca
        width_pca
        height_ionmaps
        width_ionmaps

    main:
        visualize_results(combined_metrics, main_outdir, rt_unit, output_column_order, spikein_columns, output_table_type, search_spike_ins, height_barplots, width_barplots, height_pca, width_pca, height_ionmaps, width_ionmaps)

    emit:
        jsons = visualize_results.out[0]
        htmls = visualize_results.out[1]
        tables = visualize_results.out[2]
        combined_metrics = visualize_results.out[3]
        fig15_additional_headers = visualize_results.out[5]
        fig16_BRUKER_calibrants = visualize_results.out[6]
}

process visualize_results {
    label 'mcquac_image'

    cpus 2
    memory '16.GB'

    publishDir "${main_outdir}/qc_results", mode:'copy'

    input:
    path combined_metrics
    val main_outdir
    val rt_unit
    val output_column_order
    val spikein_columns
    val output_table_type
    val search_spike_ins
    val height_barplots
    val width_barplots
    val height_pca
    val width_pca
    val height_ionmaps
    val width_ionmaps

    output:
    path("*.json"), optional: true
    path("*.html"), optional: true
    path("*.${output_table_type}"), optional: true
    path(combined_metrics), optional: true
    path("fig13_MS1_map"), optional: true
    path("fig15_additional_headers"), optional: true
    path("fig16_BRUKER_calibrants"), optional: true
    path("to_log_with_nf_later.log"), optional: true

    script:
    """
    if ${search_spike_ins}
    then 
        QC_visualization.py -hdf5_files ${combined_metrics} -output "." -spikeins -RT_unit ${rt_unit} -output_column_order ${output_column_order} -spikein_columns ${spikein_columns} -output_table_type ${output_table_type} -height_barplots ${height_barplots} -width_barplots ${width_barplots} -height_pca ${height_pca} -width_pca ${width_pca} -height_ionmaps ${height_ionmaps} -width_ionmaps ${width_ionmaps}
    else
        QC_visualization.py -hdf5_files ${combined_metrics} -output "." -RT_unit ${rt_unit} -output_column_order ${output_column_order} -spikein_columns ${spikein_columns} -output_table_type ${output_table_type} -height_barplots ${height_barplots} -width_barplots ${width_barplots} -height_pca ${height_pca} -width_pca ${width_pca} -height_ionmaps ${height_ionmaps} -width_ionmaps ${width_ionmaps}
    fi
    """
}
