questions = ['Is the animal the main subject in this photo',
    'Is the subject looking towards the camera',
    'What is the setting of this photo',
    "How far is the subject's face from the camera, in feet",
    'What color are the clothes that the subject is wearing',
    'What pose is the subject in',
    'Describe the lighting in this photo in one phrase',
    'Looking carefully at the background, is this photo taken indoors or outdoors',
    'What is in the background of the photo',
    'What are all of the items in the background of this photo']

import replicate
import os
import glob
os.environ["REPLICATE_API_TOKEN"] = "267b0b4b72042741136a411c024c18133ade115f" #Raymond's
bmodel = replicate.models.get("andreasjansson/blip-2")
bversion = bmodel.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")

def prompter(image_dir):
    answers = []
    imnames = glob.glob(image_dir+"*.jpg") + glob.glob(image_dir+"*.jpeg") + glob.glob(image_dir+"*.png")
    for imname in imnames:
        immy = open(imname, "rb")
        these_answers = []
        for p in questions:
            these_answers.append(replicate.predictions.create(
                version=bversion,
                input={
                    "image": immy,
                    "question": p
                },
            ))
    custom_prompts = []
    for these_answers in answers:
        for i, a in enumerate(these_answers):
            a.wait()
            these_answers[i] = a.output
        ifpet = these_answers[0] == 'yes'
        looking = "looking at the camera" if 'yes' in these_answers[1] else "looking away from camera"
        manwoman = these_answers[2]
        distance = these_answers[3].replace("feet", "feet away")
        clothes = "" if ifpet else " " + these_answers[4]
        custom_prompts.append(
            f"A cell phone photo of & in {these_answers[5]} wearing{clothes}, {these_answers[6]} {these_answers[7]} {manwoman}, {looking}, in the background: {answers[8]}, {answers[9]}".replace("man", "").replace("woman", "").replace("dog", "").replace("pet", "").replace("animal", "").replace("rabbit", "").replace("cat", "")
        )
        custom_prompts[-1] = custom_prompts[-1].replace("0 ft", "")
        custom_prompts[-1] = custom_prompts[-1].replace("&", "{}")
    print(custom_prompts)
    return custom_prompts