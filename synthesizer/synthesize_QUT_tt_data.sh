#!/usr/bin/env bash


#echo 'synthesizing newnoisett'
#python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_newnoise_tt.cfg
echo 'synthesizing QUT_CAFEtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_QUT_CAFE_tt.cfg
echo 'synthesizing QUT_HOMEtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_QUT_HOME_tt.cfg
# echo 'synthesizing QUT_STREETtt'
# python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_QUT_STREET_tt.cfg
# echo 'synthesizing QUT_CARtt'
# python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_QUT_CAR_tt.cfg
echo 'synthesizing QUT_REVERBtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_QUT_REVERB_tt.cfg
# echo 'synthesizing CAFtt'
# python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_CAF_tt.cfg
# echo 'synthesizing PEDtt'
# python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_PED_tt.cfg
# echo 'synthesizing STRtt'
# python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_STR_tt.cfg
# echo 'synthesizing BUStt'
# python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_BUS_tt.cfg






