import tkinter as tk

def run_quiz(root, stats_entries):
    scroll_frame = tk.Toplevel(root)
    scroll_frame.title("Personality Test")
    scroll_frame.geometry("700x600")  # set a reasonable window size

    # Create a canvas and a vertical scrollbar
    canvas = tk.Canvas(scroll_frame)
    scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    # This frame will contain all the quiz widgets
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    #Threat

    #Freq
    tk.Label(scroll_frame, text="How often did feel scared or angry during your childhood?").pack(anchor="w")
    q1_t_freq_var = tk.IntVar(value=3)
    t_freq_options = ["Not at all", "Rarely", "Sometimes", "Often", "Always"]

    #i is the index starting from 1, and option is from the options
    for i, option in enumerate(t_freq_options,1):
        #variable: where value is stored to, value: the value of the radiobutton
        tk.Radiobutton(scroll_frame, text = option, variable=q1_t_freq_var, value = i).pack(anchor="w")

    tk.Label(scroll_frame, text="How often did you meet threatening situations either from people, events"
    " (i.e. exams, applications), surroundings (unsanitary conditions, unsafe environment)").pack(anchor="w")

    q2_t_freq_var = tk.IntVar(value = 3)

    for i, option in enumerate(t_freq_options,1):
        tk.Radiobutton(scroll_frame, text = option, variable=q2_t_freq_var, value = i).pack(anchor="w")

    tk.Label(scroll_frame, text="How large or overwhelming did your childhood challenges or threat feel?").pack(anchor="w")


    q3_t_l_var = tk.IntVar(value = 3)

    t_l_options = ["Very small — barely noticeable, easy to handle", 
                   "Small — present but not very disruptive",
                   "Moderate — noticeable and requires effort to manage",
                   "Large — dominates your attention and energy",
                   "Massive — feels all-consuming or impossible to ignore"]
    
    for i, option in enumerate(t_l_options, 1):
        tk.Radiobutton(scroll_frame, text = option, variable=q3_t_l_var, value = i).pack(anchor="w")

    tk.Label(scroll_frame, text="If your biggest childhood challenge or threat were an animal, how powerful would it feel?").pack(anchor="w")

    q4_t_l_var = tk.IntVar(value = 3)

    t_l_options = ["Tiny and harmless — like a butterfly or mouse", 
                   "Small but scrappy — like a fox or wildcat",
                   "Medium strength — like a wolf or boar",
                   "Strong and formidable — like a lion or bear",
                   "Overwhelming and unstoppable — like a tiger shark or elephant"]
    
    for i, option in enumerate(t_l_options, 1):
        tk.Radiobutton(scroll_frame, text = option, variable=q4_t_l_var, value = i).pack(anchor="w")  


     # --- Threat Defensive Belonging (DB) ---
    tk.Label(scroll_frame, text="How was this threat (person, group, or situation) perceived by others by its ability to challenge others?").pack(anchor="w")

    q5_t_db_var = tk.IntVar(value=3)
    t_db_options = [
        "Easy — Others saw this as weak or inconsequential, an easy challenge.",
        "Moderately Easy — Occasionally recognized as somewhat challenging or capable but mostly overlooked.",
        "Moderately Difficult — Generally acknowledged as reasonably challenging or capable of being difficult.",
        "Very Difficult — Widely recognized as a strong 'defender' or obstacle, respected for its potential to overcome others.",
        "Legendary Difficulty — Universally perceived as extremely formidable; almost everyone recognized its challenge."
    ]

    for i, option in enumerate(t_db_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q5_t_db_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="How much did this reputation as a challenge affect how others in society behaved toward them or respected them?").pack(anchor="w")

    q6_t_db_var = tk.IntVar(value=3)
    t_db_influence_options = [
        "No influence — Their reputation as a challenge had little to no impact on how others acted.",
        "Minor influence — People occasionally acknowledged or adjusted to their challenging presence.",
        "Moderate influence — Their reputation as a challenge noticeably shaped social interactions.",
        "High influence — Others actively respected, avoided, or prepared for them because of their challenging reputation.",
        "Very high influence — Their reputation dominated interactions; people consistently measured themselves against or feared them."
    ]

    for i, option in enumerate(t_db_influence_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q6_t_db_var, value=i).pack(anchor="w")


    # --- Threat Nurturing Belonging (NB) ---
    tk.Label(scroll_frame, text="How much was this threat (person, group, or situation) perceived by others as having a nurturing or supportive role in society?").pack(anchor="w")

    q7_t_nb_var = tk.IntVar(value=3)
    t_nb_options = [
        "Not at all — Seen as cold or unsupportive.",
        "Slightly — Occasionally seen as caring or helpful.",
        "Moderately — Frequently regarded as nurturing or supportive.",
        "Highly — Widely respected for their nurturing qualities.",
        "Extremely — Their identity was strongly defined by being a nurturer."
    ]

    for i, option in enumerate(t_nb_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q7_t_nb_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="How much did this reputation as a nurturer affect how others in society behaved toward them or interacted with them?").pack(anchor="w")

    q8_t_nb_var = tk.IntVar(value=3)
    t_nb_influence_options = [
        "No influence — Their nurturing reputation had little effect on others.",
        "Minor influence — Others occasionally responded with trust or openness.",
        "Moderate influence — Their nurturing reputation noticeably shaped interactions.",
        "High influence — People often deferred, trusted, or relied on them due to this reputation.",
        "Very high influence — Their nurturing role dominated interactions; others consistently trusted or depended on them."
    ]

    for i, option in enumerate(t_nb_influence_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q8_t_nb_var, value=i).pack(anchor="w")

        # --- Frequency (Q9, Q10) ---
    tk.Label(scroll_frame, text="How often did you set or pursue personal goals during your childhood?").pack(anchor="w")

    q9_freq_var = tk.IntVar(value=3)
    freq_options = [
        "Not at all — You rarely set or focused on any goals.",
        "Rarely — Goals were occasional and not very frequent.",
        "Sometimes — You pursued goals with moderate regularity.",
        "Often — Setting and pursuing goals was a frequent part of your life.",
        "Always — Goals were a consistent and central part of your daily life."
    ]

    for i, option in enumerate(freq_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q9_freq_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="How often did you engage in goals that felt challenging or rewarding during your childhood?").pack(anchor="w")

    q10_freq_var = tk.IntVar(value=3)
    freq_challenge_options = [
        "Not at all — You rarely pursued goals that required effort or brought satisfaction.",
        "Rarely — Challenging or rewarding goals happened occasionally.",
        "Sometimes — You engaged in them with moderate frequency.",
        "Often — Such goals were a common part of your experiences.",
        "Always — You frequently pursued goals that felt meaningful, challenging, or rewarding."
    ]

    for i, option in enumerate(freq_challenge_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q10_freq_var, value=i).pack(anchor="w")


    # --- Livelihood (Q11, Q12) ---
    tk.Label(scroll_frame, text="As a child, when you thought about your goals or achievements, how substantial or 'big' did they feel compared to smaller everyday aims?").pack(anchor="w")

    q11_livelihood_var = tk.IntVar(value=3)
    livelihood_size_options = [
        "Very small — Felt trivial, like chasing after scraps.",
        "Small — Noticeable but not very significant.",
        "Moderate — Clearly worthwhile, but not overwhelming.",
        "Large — Substantial and demanding, required real effort.",
        "Massive — Felt like a 'big prize,' highly significant and defining."
    ]

    for i, option in enumerate(livelihood_size_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q11_livelihood_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="If you succeeded as a child in reaching your childhood goals, how rewarding or satisfying would it feel, compared to everyday successes?").pack(anchor="w")

    q12_livelihood_var = tk.IntVar(value=3)
    livelihood_reward_options = [
        "Minimal — Hardly rewarding, almost not worth the effort.",
        "Low — Slightly rewarding, but not very motivating.",
        "Moderate — Noticeably rewarding, gave a sense of accomplishment.",
        "High — Strongly rewarding, felt deeply satisfying.",
        "Exceptional — Extremely rewarding, like earning a big accomplishment or ultimate prize."
    ]

    for i, option in enumerate(livelihood_reward_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q12_livelihood_var, value=i).pack(anchor="w")


    # --- Defensive Belonging (Q13, Q14) ---
    tk.Label(scroll_frame, text="How difficult or formidable would others have perceived this goal to be for you as a child?").pack(anchor="w")

    q13_db_var = tk.IntVar(value=3)
    db_challenge_options = [
        "Very easy — Others likely thought it was trivial or effortless.",
        "Somewhat easy — Occasionally recognized as challenging, but mostly routine.",
        "Moderate — Generally acknowledged as a noticeable challenge.",
        "Very difficult — Widely seen as demanding, requiring effort and skill.",
        "Extremely formidable — Universally recognized as a tough or prestigious goal, commanding respect."
    ]

    for i, option in enumerate(db_challenge_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q13_db_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="If you successfully accomplished this goal as a child, how much did you expect others to respect you for being able to tackle such challenging tasks?").pack(anchor="w")

    q14_db_var = tk.IntVar(value=3)
    db_influence_options = [
        "No influence — Achieving the goal didn’t make others take notice.",
        "Minor influence — People occasionally recognized your ability to handle difficult challenges.",
        "Moderate influence — Success noticeably changed how others interacted with or regarded you.",
        "High influence — Others actively respected or deferred to you for being capable of overcoming such challenges.",
        "Very high influence — Your achievement dominated social perception; people consistently admired or measured themselves against you."
    ]

    for i, option in enumerate(db_influence_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q14_db_var, value=i).pack(anchor="w")


    # --- Nurturing Belonging (Q15) ---
    tk.Label(scroll_frame, text="If you accomplished this goal as a child, how much would it have influenced how others treated, trusted, or relied on you because the goal was supportive or helpful to others?").pack(anchor="w")

    q15_nb_var = tk.IntVar(value=3)
    nb_influence_options = [
        "No influence — Achieving the goal didn’t change how others interacted with or trusted you for being supportive.",
        "Minor influence — People occasionally acknowledged or appreciated your effort to be helpful.",
        "Moderate influence — Your success noticeably affected social interactions or trust in being helpful.",
        "High influence — Others actively deferred to, trusted, or sought your help because of this accomplishment of being helpful.",
        "Very high influence — Your achievement strongly shaped social interactions; others consistently relied on, respected, or admired you for being helpful."
    ]

    for i, option in enumerate(nb_influence_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q15_nb_var, value=i).pack(anchor="w")

        # --- Nurturing Belonging (Q16) ---
    tk.Label(scroll_frame, text="How much was this goal perceived by others as being helpful, supportive, or beneficial to others?").pack(anchor="w")

    q16_nb_var = tk.IntVar(value=3)

    nb_perception_options = [
        "Not at all — Others didn’t see this goal as helping anyone.",
        "Slightly — Occasionally recognized as somewhat helpful or supportive.",
        "Moderately — Frequently seen as beneficial or contributing to others’ well-being.",
        "Highly — Widely regarded as a goal that meaningfully supported or improved others’ lives.",
        "Extremely — Your goal was clearly identified as significantly helpful or nurturing; others strongly valued it."
    ]

    for i, option in enumerate(nb_perception_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q16_nb_var, value=i).pack(anchor="w")

        # --- Allies Frequency ---
    tk.Label(scroll_frame, text="During your childhood, how often did you encounter supportive figures (e.g. kind teacher, friends, or caring family members) who made you feel protected or encouraged?").pack(anchor="w")

    q17_a_freq_var = tk.IntVar(value=3)
    a_freq_options = [
        "Almost never — I rarely had such supportive figures.",
        "Occasionally — A few supportive figures came into my life from time to time.",
        "Sometimes — I encountered supportive figures with moderate frequency.",
        "Often — I had many supportive figures who appeared throughout my childhood.",
        "Very often — Supportive figures were a consistent and reliable part of my childhood."
    ]
    for i, option in enumerate(a_freq_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q17_a_freq_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="How frequently during your childhood did you spend time with supportive people (teachers, peers, relatives, strangers, etc.) who could help and/or make you feel cared for?").pack(anchor="w")

    q18_a_freq_var = tk.IntVar(value=3)
    a_freq_time_options = [
        "Almost never — I rarely spent time with supportive people.",
        "Rarely — I occasionally spent time with supportive people.",
        "Sometimes — I spent time with supportive people with moderate frequency.",
        "Often — I frequently spent time with supportive people.",
        "Very often — Supportive people were a consistent and regular part of my life."
    ]
    for i, option in enumerate(a_freq_time_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q18_a_freq_var, value=i).pack(anchor="w")


    # --- Allies Livelihood ---
    tk.Label(scroll_frame, text="How strong and physically capable did the supportive people in your childhood (teachers, friends, relatives, etc.) seem to you?").pack(anchor="w")

    q19_a_l_var = tk.IntVar(value=3)
    a_l_strength_options = [
        "Very weak — seemed fragile or unable to help much.",
        "Weak — they could help a little, but not in big ways.",
        "Moderate — they had average strength and capability.",
        "Strong — they seemed reliable and capable of helping.",
        "Very strong — they seemed powerful, healthy, and consistently able to protect or support you."
    ]
    for i, option in enumerate(a_l_strength_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q19_a_l_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="If your most supportive childhood ally were an animal, how powerful would it feel?").pack(anchor="w")

    q20_a_l_var = tk.IntVar(value=3)
    a_l_animal_options = [
        "Tiny and fragile — like a sparrow or rabbit.",
        "Small but sturdy — like a dog or goat.",
        "Moderately strong — like a horse or deer.",
        "Strong and protective — like a lion or ox.",
        "Overwhelmingly powerful — like an elephant or eagle."
    ]
    for i, option in enumerate(a_l_animal_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q20_a_l_var, value=i).pack(anchor="w")


    # --- Allies Defensive Belonging ---
    tk.Label(scroll_frame, text="How was this ally (person, group, or situation) perceived by others in terms of their ability to protect or defend those around them?").pack(anchor="w")

    q21_a_db_var = tk.IntVar(value=3)
    a_db_options = [
        "Weak protector — Others saw them as unable to defend or support much.",
        "Somewhat protective — Occasionally recognized as offering some defense or reliability.",
        "Moderately protective — Generally acknowledged as capable of protecting or standing up for others.",
        "Strong protector — Widely respected as someone dependable and reliable in defending others.",
        "Legendary protector — Universally perceived as an exceptional defender; others consistently trusted in or relied upon their protective role."
    ]
    for i, option in enumerate(a_db_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q21_a_db_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="How much did this reputation as a protector or defender affect how others in society behaved toward them or respected them?").pack(anchor="w")

    q22_a_db_var = tk.IntVar(value=3)
    a_db_influence_options = [
        "No influence — Their reputation as a protector had little to no effect on others’ behavior.",
        "Minor influence — People occasionally acknowledged or deferred to their protective presence.",
        "Moderate influence — Their reputation noticeably shaped how others interacted with or relied on them.",
        "High influence — Others actively respected, trusted, or turned to them because of their protective reputation.",
        "Very high influence — Their reputation dominated social interactions; people consistently sought their support or relied on them for defense."
    ]
    for i, option in enumerate(a_db_influence_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q22_a_db_var, value=i).pack(anchor="w")


    # --- Allies Nurturing Belonging ---
    tk.Label(scroll_frame, text="How much was this ally (person, group, or situation) perceived by others as having a nurturing or supportive role in society?").pack(anchor="w")

    q23_a_nb_var = tk.IntVar(value=3)
    a_nb_options = [
        "Not at all — Seen as indifferent or unsupportive.",
        "Slightly — Occasionally noticed for being helpful or kind.",
        "Moderately — Frequently regarded as nurturing and supportive.",
        "Highly — Widely respected for their caring or uplifting nature.",
        "Extremely — Strongly defined by their nurturing role; almost everyone saw them as a source of support."
    ]
    for i, option in enumerate(a_nb_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q23_a_nb_var, value=i).pack(anchor="w")

    tk.Label(scroll_frame, text="How much did this reputation as a nurturer affect how others in society behaved toward them or interacted with them?").pack(anchor="w")

    q24_a_nb_var = tk.IntVar(value=3)
    a_nb_influence_options = [
        "No influence — Their nurturing reputation had little effect on others.",
        "Minor influence — Others occasionally responded with openness or reliance.",
        "Moderate influence — Their nurturing reputation noticeably shaped social interactions.",
        "High influence — People often trusted, deferred to, or sought them out for support.",
        "Very high influence — Their nurturing role dominated interactions; others consistently depended on them for care or guidance."
    ]
    for i, option in enumerate(a_nb_influence_options, 1):
        tk.Radiobutton(scroll_frame, text=option, variable=q24_a_nb_var, value=i).pack(anchor="w")


    def finish_quiz():
        t_freq_var = q1_t_freq_var.get() + q2_t_freq_var.get()
        a_freq_var = q17_a_freq_var.get() + q18_a_freq_var.get()
        p_freq_var = q9_freq_var.get() + q10_freq_var.get()

        total_freq_var = t_freq_var + a_freq_var + p_freq_var

        t_freq = t_freq_var / total_freq_var
        a_freq = a_freq_var / total_freq_var
        p_freq = 1 - t_freq - a_freq

        stats_entries["prob_threat"].delete(0, tk.END)
        stats_entries["prob_threat"].insert(0, str(t_freq))

        stats_entries["prob_ally"].delete(0, tk.END)
        stats_entries["prob_ally"].insert(0, str(a_freq))

        stats_entries["prob_prey"].delete(0, tk.END)
        stats_entries["prob_prey"].insert(0, str(p_freq))

        l_threat_level = int((q3_t_l_var.get() + q4_t_l_var.get()) * 100 / 8)
        l_ally_level = int((q19_a_l_var.get() + q20_a_l_var.get()) * 100 / 8)
        l_prey_level = int((q11_livelihood_var.get() + q12_livelihood_var.get()) * 100 / 8)

        db_threat_level = int((q5_t_db_var.get() + q6_t_db_var.get()) * 100 / 8)
        db_ally_level = int((q21_a_db_var.get() + q22_a_db_var.get()) * 100 / 8)
        db_prey_level = int((q13_db_var.get() + q14_db_var.get()) * 100 / 8)

        nb_threat_level = int((q7_t_nb_var.get() + q8_t_nb_var.get()) * 100 / 8)
        nb_ally_level = int((q23_a_nb_var.get() + q24_a_nb_var.get()) * 100 / 8)
        nb_prey_level = int((q15_nb_var.get() + q16_nb_var.get()) * 100 / 8)


        stats_entries["tLowerSitL"].delete(0, tk.END)
        stats_entries["tLowerSitL"].insert(0, str(l_threat_level - 10))
        stats_entries["tHigherSitL"].delete(0, tk.END)
        stats_entries["tHigherSitL"].insert(0, str(l_threat_level + 10))
        stats_entries["aLowerSitL"].delete(0, tk.END)
        stats_entries["aLowerSitL"].insert(0, str(l_ally_level - 10))
        stats_entries["aHigherSitL"].delete(0, tk.END)
        stats_entries["aHigherSitL"].insert(0, str(l_ally_level + 10))
        stats_entries["pLowerSitL"].delete(0, tk.END)
        stats_entries["pLowerSitL"].insert(0, str(l_prey_level - 10))
        stats_entries["pHigherSitL"].delete(0, tk.END)
        stats_entries["pHigherSitL"].insert(0, str(l_prey_level + 10))

        stats_entries["tLowerSitDB"].delete(0, tk.END)
        stats_entries["tLowerSitDB"].insert(0, str(db_threat_level - 10))
        stats_entries["tHigherSitDB"].delete(0, tk.END)
        stats_entries["tHigherSitDB"].insert(0, str(db_threat_level + 10))
        stats_entries["aLowerSitDB"].delete(0, tk.END)
        stats_entries["aLowerSitDB"].insert(0, str(db_ally_level - 10))
        stats_entries["aHigherSitDB"].delete(0, tk.END)
        stats_entries["aHigherSitDB"].insert(0, str(db_ally_level + 10))
        stats_entries["pLowerSitDB"].delete(0, tk.END)
        stats_entries["pLowerSitDB"].insert(0, str(db_prey_level - 10))
        stats_entries["pHigherSitDB"].delete(0, tk.END)
        stats_entries["pHigherSitDB"].insert(0, str(db_prey_level + 10))

        stats_entries["tLowerSitNB"].delete(0, tk.END)
        stats_entries["tLowerSitNB"].insert(0, str(nb_threat_level - 10))
        stats_entries["tHigherSitNB"].delete(0, tk.END)
        stats_entries["tHigherSitNB"].insert(0, str(nb_threat_level + 10))
        stats_entries["aLowerSitNB"].delete(0, tk.END)
        stats_entries["aLowerSitNB"].insert(0, str(nb_ally_level - 10))
        stats_entries["aHigherSitNB"].delete(0, tk.END)
        stats_entries["aHigherSitNB"].insert(0, str(nb_ally_level + 10))
        stats_entries["pLowerSitNB"].delete(0, tk.END)
        stats_entries["pLowerSitNB"].insert(0, str(nb_prey_level - 10))
        stats_entries["pHigherSitNB"].delete(0, tk.END)
        stats_entries["pHigherSitNB"].insert(0, str(nb_prey_level + 10))

        scroll_frame.destroy()

    tk.Button(scroll_frame, text = "Finish Test", command=finish_quiz).pack()