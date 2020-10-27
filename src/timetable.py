Jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
Dic   = {"Lundi":1, "Mardi":2, "Mercredi":3, "Jeudi":4, "Vendredi":5, "Samedi":6, "Dimanche":7}

w = 700
h = 500
shift = 3
x = w/8
y = h/19

ratio_x = x/(w+2*shift)
ratio_y = y/(h+2*shift)

s = 1
m = s * 60
h = m * 60

from tkinter import Tk, Canvas

def init_timetable(first):
    root = Tk()
    root.resizable(False, True)

    canvas = Canvas(root, width = w+2*shift, height = h+2*shift, background="white")
    canvas.pack(fill="both", expand=True)

    #grille
    for i in range(8):
        for j in range(18):
            canvas.create_rectangle((x*i+shift,y*j+shift), (x*(i+1)+shift,y*(j+1)+shift))

    #jours
    for i in range(len(Jours)) :
        mylabel = canvas.create_text((shift+1.5*x + x*i, shift+0.5*y), text=Jours[i], fill = "black", font=("Arial", 10))

    #horaires
    for i in range(6, 23):
        label = ""
        if (i < 10):
            label = "0" + str(i)
        else:
            label = str(i)

        mylabel = canvas.create_text((shift+0.5*x, shift +1.5*y + (i-6)*y), text=label +":00", fill = "black", font=("Arial", 10))



    def transformation(x):
        return (x - (6*h)) / (h) + 1


    def add_rectangle(jour, heure, duree, nom, couleur = "green"):
        ys = transformation(heure) * y
        ye = transformation(heure + duree) * y



        if (jour == "All"):
            xs = x
            xe = 8*x



        else:
            xs = Dic[jour] * x
            xe = Dic[jour] * x + x


        canvas.create_rectangle((xs+shift,ys+shift), (xe+shift,ye+shift), fill=couleur)
        canvas.create_text(((xs+xe)/2, (ys+ye)/2), text=nom, fill = "white", font=("Arial", 12))


    #pas de machine
    add_rectangle("All", 6*h, 4*h, "Machine occupée", "red")
    add_rectangle("All", 20*h, 3*h, "Machine occupée", "red")
    add_rectangle("Dimanche", 6*h, 17*h, "", "red")

    #create task
    curr = first


    if (not curr.disp):
        add_rectangle(curr.jour, curr.heure, curr.duree, curr.name)

    curr = curr.next

    while (curr != first):
        if (not curr.disp and curr.name not in Jours):
            add_rectangle(curr.jour, curr.heure, curr.duree, curr.name)
        curr = curr.next

    root.mainloop()
