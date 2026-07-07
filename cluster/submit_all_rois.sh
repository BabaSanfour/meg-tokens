#!/bin/bash

# ==============================================================================
# Unified Master SLURM Submitter for All HCPMMP1 ROIs
# Replaces all 12 legacy `run_all_classif_*.sh` scripts.
# ==============================================================================
# Usage Example:
# export TOKENS_BIDS=/path/to/tokens-bids
# ./submit_all_rois.sh
# ==============================================================================

TOKENS_BIDS=${TOKENS_BIDS:?Set TOKENS_BIDS to the BIDS derivatives root}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: TOKENS_BIDS=/path/to/tokens-bids ./submit_all_rois.sh"
    echo ""
    echo "This script iterates over all 360 HCPMMP1 ROIs and submits them to the SLURM cluster"
    echo "using the unified cluster/job_decoding.sh script."
    echo ""
    echo "Optional environment variables: CONDITIONS, ALIGN_TO, SOURCE_METHOD, PARC, PERMUTATIONS."
    exit 1
fi

ROIS="L_10d_ROI-lh L_10pp_ROI-lh L_10r_ROI-lh L_10v_ROI-lh L_11l_ROI-lh L_13l_ROI-lh L_1_ROI-lh L_23c_ROI-lh L_23d_ROI-lh L_24dd_ROI-lh L_24dv_ROI-lh L_25_ROI-lh L_2_ROI-lh L_31a_ROI-lh L_31pd_ROI-lh L_31pv_ROI-lh L_33pr_ROI-lh L_3a_ROI-lh L_3b_ROI-lh L_43_ROI-lh L_44_ROI-lh L_45_ROI-lh L_46_ROI-lh L_47l_ROI-lh L_47m_ROI-lh L_47s_ROI-lh L_4_ROI-lh L_52_ROI-lh L_55b_ROI-lh L_5L_ROI-lh L_5m_ROI-lh L_5mv_ROI-lh L_6a_ROI-lh L_6d_ROI-lh L_6ma_ROI-lh L_6mp_ROI-lh L_6r_ROI-lh L_6v_ROI-lh L_7AL_ROI-lh L_7Am_ROI-lh L_7PC_ROI-lh L_7PL_ROI-lh L_7Pm_ROI-lh L_7m_ROI-lh L_8Ad_ROI-lh L_8Av_ROI-lh L_8BL_ROI-lh L_8BM_ROI-lh L_8C_ROI-lh L_9-46d_ROI-lh L_9a_ROI-lh L_9m_ROI-lh L_9p_ROI-lh L_A1_ROI-lh L_A4_ROI-lh L_A5_ROI-lh L_AAIC_ROI-lh L_AIP_ROI-lh L_AVI_ROI-lh L_DVT_ROI-lh L_EC_ROI-lh L_FEF_ROI-lh L_FFC_ROI-lh L_FOP1_ROI-lh L_FOP2_ROI-lh L_FOP3_ROI-lh L_FOP4_ROI-lh L_FOP5_ROI-lh L_FST_ROI-lh L_H_ROI-lh L_IFJa_ROI-lh L_IFJp_ROI-lh L_IFSa_ROI-lh L_IFSp_ROI-lh L_IP0_ROI-lh L_IP1_ROI-lh L_IP2_ROI-lh L_IPS1_ROI-lh L_Ig_ROI-lh L_LBelt_ROI-lh L_LIPd_ROI-lh L_LIPv_ROI-lh L_LO1_ROI-lh L_LO2_ROI-lh L_LO3_ROI-lh L_MBelt_ROI-lh L_MIP_ROI-lh L_MI_ROI-lh L_MST_ROI-lh L_MT_ROI-lh L_OFC_ROI-lh L_OP1_ROI-lh L_OP2-3_ROI-lh L_OP4_ROI-lh L_PBelt_ROI-lh L_PCV_ROI-lh L_PEF_ROI-lh L_PF_ROI-lh L_PFcm_ROI-lh L_PFm_ROI-lh L_PFop_ROI-lh L_PFt_ROI-lh L_PGi_ROI-lh L_PGp_ROI-lh L_PGs_ROI-lh L_PHA1_ROI-lh L_PHA2_ROI-lh L_PHA3_ROI-lh L_PHT_ROI-lh L_PH_ROI-lh L_PIT_ROI-lh L_PI_ROI-lh L_POS1_ROI-lh L_POS2_ROI-lh L_PSL_ROI-lh L_PeEc_ROI-lh L_Pir_ROI-lh L_PoI1_ROI-lh L_PoI2_ROI-lh L_PreS_ROI-lh L_ProS_ROI-lh L_RI_ROI-lh L_RSC_ROI-lh L_SCEF_ROI-lh L_SFL_ROI-lh L_STGa_ROI-lh L_STSda_ROI-lh L_STSdp_ROI-lh L_STSva_ROI-lh L_STSvp_ROI-lh L_STV_ROI-lh L_TA2_ROI-lh L_TE1a_ROI-lh L_TE1m_ROI-lh L_TE1p_ROI-lh L_TE2a_ROI-lh L_TE2p_ROI-lh L_TF_ROI-lh L_TGd_ROI-lh L_TGv_ROI-lh L_TPOJ1_ROI-lh L_TPOJ2_ROI-lh L_TPOJ3_ROI-lh L_V1_ROI-lh L_V2_ROI-lh L_V3A_ROI-lh L_V3B_ROI-lh L_V3CD_ROI-lh L_V3_ROI-lh L_V4_ROI-lh L_V4t_ROI-lh L_V6A_ROI-lh L_V6_ROI-lh L_V7_ROI-lh L_V8_ROI-lh L_VIP_ROI-lh L_VMV1_ROI-lh L_VMV2_ROI-lh L_VMV3_ROI-lh L_VVC_ROI-lh L_a10p_ROI-lh L_a24_ROI-lh L_a24pr_ROI-lh L_a32pr_ROI-lh L_a47r_ROI-lh L_a9-46v_ROI-lh L_d23ab_ROI-lh L_d32_ROI-lh L_i6-8_ROI-lh L_p10p_ROI-lh L_p24_ROI-lh L_p24pr_ROI-lh L_p32_ROI-lh L_p32pr_ROI-lh L_p47r_ROI-lh L_p9-46v_ROI-lh L_pOFC_ROI-lh L_s32_ROI-lh L_s6-8_ROI-lh L_v23ab_ROI-lh R_10d_ROI-rh R_10pp_ROI-rh R_10r_ROI-rh R_10v_ROI-rh R_11l_ROI-rh R_13l_ROI-rh R_1_ROI-rh R_23c_ROI-rh R_23d_ROI-rh R_24dd_ROI-rh R_24dv_ROI-rh R_25_ROI-rh R_2_ROI-rh R_31a_ROI-rh R_31pd_ROI-rh R_31pv_ROI-rh R_33pr_ROI-rh R_3a_ROI-rh R_3b_ROI-rh R_43_ROI-rh R_44_ROI-rh R_45_ROI-rh R_46_ROI-rh R_47l_ROI-rh R_47m_ROI-rh R_47s_ROI-rh R_4_ROI-rh R_52_ROI-rh R_55b_ROI-rh R_5L_ROI-rh R_5m_ROI-rh R_5mv_ROI-rh R_6a_ROI-rh R_6d_ROI-rh R_6ma_ROI-rh R_6mp_ROI-rh R_6r_ROI-rh R_6v_ROI-rh R_7AL_ROI-rh R_7Am_ROI-rh R_7PC_ROI-rh R_7PL_ROI-rh R_7Pm_ROI-rh R_7m_ROI-rh R_8Ad_ROI-rh R_8Av_ROI-rh R_8BL_ROI-rh R_8BM_ROI-rh R_8C_ROI-rh R_9-46d_ROI-rh R_9a_ROI-rh R_9m_ROI-rh R_9p_ROI-rh R_A1_ROI-rh R_A4_ROI-rh R_A5_ROI-rh R_AAIC_ROI-rh R_AIP_ROI-rh R_AVI_ROI-rh R_DVT_ROI-rh R_EC_ROI-rh R_FEF_ROI-rh R_FFC_ROI-rh R_FOP1_ROI-rh R_FOP2_ROI-rh R_FOP3_ROI-rh R_FOP4_ROI-rh R_FOP5_ROI-rh R_FST_ROI-rh R_H_ROI-rh R_IFJa_ROI-rh R_IFJp_ROI-rh R_IFSa_ROI-rh R_IFSp_ROI-rh R_IP0_ROI-rh R_IP1_ROI-rh R_IP2_ROI-rh R_IPS1_ROI-rh R_Ig_ROI-rh R_LBelt_ROI-rh R_LIPd_ROI-rh R_LIPv_ROI-rh R_LO1_ROI-rh R_LO2_ROI-rh R_LO3_ROI-rh R_MBelt_ROI-rh R_MIP_ROI-rh R_MI_ROI-rh R_MST_ROI-rh R_MT_ROI-rh R_OFC_ROI-rh R_OP1_ROI-rh R_OP2-3_ROI-rh R_OP4_ROI-rh R_PBelt_ROI-rh R_PCV_ROI-rh R_PEF_ROI-rh R_PF_ROI-rh R_PFcm_ROI-rh R_PFm_ROI-rh R_PFop_ROI-rh R_PFt_ROI-rh R_PGi_ROI-rh R_PGp_ROI-rh R_PGs_ROI-rh R_PHA1_ROI-rh R_PHA2_ROI-rh R_PHA3_ROI-rh R_PHT_ROI-rh R_PH_ROI-rh R_PIT_ROI-rh R_PI_ROI-rh R_POS1_ROI-rh R_POS2_ROI-rh R_PSL_ROI-rh R_PeEc_ROI-rh R_Pir_ROI-rh R_PoI1_ROI-rh R_PoI2_ROI-rh R_PreS_ROI-rh R_ProS_ROI-rh R_RI_ROI-rh R_RSC_ROI-rh R_SCEF_ROI-rh R_SFL_ROI-rh R_STGa_ROI-rh R_STSda_ROI-rh R_STSdp_ROI-rh R_STSva_ROI-rh R_STSvp_ROI-rh R_STV_ROI-rh R_TA2_ROI-rh R_TE1a_ROI-rh R_TE1m_ROI-rh R_TE1p_ROI-rh R_TE2a_ROI-rh R_TE2p_ROI-rh R_TF_ROI-rh R_TGd_ROI-rh R_TGv_ROI-rh R_TPOJ1_ROI-rh R_TPOJ2_ROI-rh R_TPOJ3_ROI-rh R_V1_ROI-rh R_V2_ROI-rh R_V3A_ROI-rh R_V3B_ROI-rh R_V3CD_ROI-rh R_V3_ROI-rh R_V4_ROI-rh R_V4t_ROI-rh R_V6A_ROI-rh R_V6_ROI-rh R_V7_ROI-rh R_V8_ROI-rh R_VIP_ROI-rh R_VMV1_ROI-rh R_VMV2_ROI-rh R_VMV3_ROI-rh R_VVC_ROI-rh R_a10p_ROI-rh R_a24_ROI-rh R_a24pr_ROI-rh R_a32pr_ROI-rh R_a47r_ROI-rh R_a9-46v_ROI-rh R_d23ab_ROI-rh R_d32_ROI-rh R_i6-8_ROI-rh R_p10p_ROI-rh R_p24_ROI-rh R_p24pr_ROI-rh R_p32_ROI-rh R_p32pr_ROI-rh R_p47r_ROI-rh R_p9-46v_ROI-rh R_pOFC_ROI-rh R_s32_ROI-rh R_s6-8_ROI-rh R_v23ab_ROI-rh"

for roi in $ROIS; do
    echo "Submitting decoding job for ROI: $roi"
    sbatch cluster/job_decoding.sh "$roi"
done

echo "All 360 ROI jobs submitted to the cluster!"
