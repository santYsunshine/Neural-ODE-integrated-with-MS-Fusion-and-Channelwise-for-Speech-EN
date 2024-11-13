#!/usr/bin/env bash

# /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr=1
# gen_scpfolder=2
# gen_scp(){
#     if [ -d 2 ]; then
#         if [ -f 3 ]; then
#             rm 3
#         fi
#         sh 1/gen_scp.sh 2 3
#     fi
# }


echo 'start generate .scp in tt'
# if [ -d /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix ]; then
#     if [ -f /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp ]; then
#         rm /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
#     fi
#     sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
# fi
# sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
# sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk1.scp
# sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk2.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk1.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk2.scp

#echo 'start generate .scp in birdtt'
# if [ -d /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix ]; then
#     if [ -f /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp ]; then
#         rm /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
#     fi
#     sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
# fi
# sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/mix.scp
# sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk1.scp
# sh /asr_denoiser/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/tt/spk2.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/birdtt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/birdtt/mix.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/birdtt/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/birdtt/spk1.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/birdtt/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/birdtt/spk2.scp

#echo 'start generate .scp in doortt'
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/doortt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/doortt/mix.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/doortt/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/doortt/spk1.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/doortt/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/doortt/spk2.scp

#echo 'start generate .scp in drumtt'
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/drumtt/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/drumtt/mix.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/drumtt/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/drumtt/spk1.scp
#sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/drumtt/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/drumtt/spk2.scp



echo 'start generate .scp in CAFtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/CAF_Vol3/CAFtt_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/CAF_Vol3/CAFtt_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/CAF_Vol3/CAFtt_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/CAF_Vol3/CAFtt_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/CAF_Vol3/CAFtt_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/CAF_Vol3/CAFtt_0dB/spk2.scp

echo 'start generate .scp in PEDtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/PED_Vol3/PEDtt_0dB/spk2.scp

echo 'start generate .scp in STRtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/STR_Vol3/STRtt_0dB/spk2.scp

echo 'start generate .scp in BUStt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/BUS_Vol3/BUStt_0dB/spk2.scp


# echo 'start generate .scp in QUTtt'
# sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUT/QUTttnew_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUT/QUTttnew_0dB/mix.scp
# sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUT/QUTttnew_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUT/QUTttnew_0dB/spk1.scp
# sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUT/QUTttnew_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUT/QUTttnew_0dB/spk2.scp

echo 'start generate .scp in QUTSTREETtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTSTREET_Vol3/QUTSTREETttnew_0dB/spk2.scp

echo 'start generate .scp in QUTREVERBtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTREVERB_Vol3/QUTREVERBttnew_0dB/spk2.scp

echo 'start generate .scp in QUTHOMEtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTHOME_Vol3/QUTHOMEttnew_0dB/spk2.scp

echo 'start generate .scp in QUTCARtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAR_Vol3/QUTCARttnew_0dB/spk2.scp

echo 'start generate .scp in QUTCAFEtt'
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_0dB/mix /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_0dB/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_0dB/s1 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_0dB/spk1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_0dB/s2 /media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/QUTCAFE_Vol3/QUTCAFEttnew_0dB/spk2.scp
