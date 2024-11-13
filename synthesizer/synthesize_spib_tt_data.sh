#!/usr/bin/env bash
echo 'synthesizing babble_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_babble_tt.cfg
echo 'synthesizing factory_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_factory_tt.cfg
echo 'synthesizing hfchannel_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_hfchannel_tt.cfg
echo 'synthesizing machinegun_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_machinegun_tt.cfg
echo 'synthesizing pink_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_pink_tt.cfg
echo 'synthesizing white_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_white_tt.cfg
echo 'synthesizing destroyerengine_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_destroyerengine_tt.cfg
echo 'synthesizing volvo_tt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_volvo_tt.cfg






