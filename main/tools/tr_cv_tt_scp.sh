#!/usr/bin/env bash


# For Training SCP file generation
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tr/mix /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tr/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tr/s1 /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tr/s1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tr/s2 /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tr/s2.scp


# For Validation SCP file generation
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/cv/mix /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/cv/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/cv/s1 /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/cv/s1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/cv/s2 /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/cv/s2.scp


# For Testing SCP file generation
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tt/mix /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tt/mix.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tt/s1 /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tt/s1.scp
sh /media/speech70809/Data01/speech_donoiser_new/main/gen_scp.sh /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tt/s2 /media/speech70809/Data01/speech_denoiser/Dataset/ner-300hr/tt/s2.scp