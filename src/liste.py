class tache:
    def __init__(self, heure, duree, disp, name, jour):
        self.heure = heure
        self.duree = duree
        self.disp  = disp
        self.next  = None
        self.name  = name
        self.jour  = jour

Jours = {0:"Lundi", 1:"Mardi", 2:"Mercredi", 3:"Jeudi", 4:"Vendredi", 5:"Samedi", 6:"Dimanche"}
Disponible = {}

s = 1
m = s * 60
h = m * 60

prev = None
for i in Jours:
  new = tache(10*h, 10*h, True, Jours[i], Jours[i])
  if prev:
    prev.next = new

  Disponible[Jours[i]] = new
  prev = new


Disponible["Dimanche"].next = Disponible["Lundi"]
Disponible["Dimanche"].disp = False

def secToHMS(nb_sec):
  q,s=divmod(nb_sec,60)
  h,m=divmod(q,60)
  return "%2d:%2d:%2d" %(h,m,s)


def DisplayAll():
  first = Disponible["Lundi"]
  curr = first

  while(curr.next != first):
    print('{}\t\t{}\t\t{}\t\t{}'.format(curr.name, secToHMS(curr.heure), secToHMS(curr.duree),curr.disp))
    #print(curr.name, secToHMS(curr.heure), "  ", secToHMS(curr.duree), "  ", curr.disp)
    curr = curr.next


  print('{}\t{}\t{}\t{}'.format(curr.name, secToHMS(curr.heure), secToHMS(curr.duree),curr.disp))

def  FindFirst(jour, heure):
  curr = Disponible[jour]

  if (not curr.next):
    return curr

  while (curr.jour == jour):

    if (curr.heure + curr.duree >= heure):
      return curr, True

    curr = curr.next
  return curr, False

def FindLocation(first, heure, duree, name, jour, bool):
  curr = first
  day  = jour

  if (bool and curr.disp == True and curr.duree > 0 and heure + duree <= curr.heure + curr.duree):
    return curr, tache(heure, duree, False, name, curr.jour)

  if (not bool and curr.disp == True and curr.duree > 0 and duree <= curr.duree):
    return curr, tache(curr.heure, duree, False, name, curr.jour)

  curr = curr.next

  while (curr != first):
    if (curr.disp == True and curr.duree > 0 and duree <= curr.duree):
      return curr, tache(curr.heure, duree, False, name, curr.jour)

    curr = curr.next





def InsertAfter(creneau, new):
  new.next     = creneau.next
  creneau.next = new


def FindPrev(creneau):
  first = Disponible["Lundi"]
  while (first.next):
    if (first.next == creneau):
      return first
    first = first.next

def UpdateInside(creneau, new):
  InsertAfter(creneau, new)

  if (new.heure + new.duree < creneau.heure+creneau.duree):
    end_new  = new.heure+new.duree
    end_cren = creneau.heure+creneau.duree

    new_2 = tache(end_new, end_cren-end_new, True, creneau.name, creneau.jour)

    InsertAfter(new, new_2)

  creneau.duree = new.heure - creneau.heure

  if (creneau.duree == 0 and creneau != Disponible["Lundi"]):
    prev = FindPrev(creneau)
    prev.next = creneau.next
    del creneau

 # DisplayAll()



def nouvelle_tache(heure, duree, nom, jour):
    first, bool = FindFirst(jour, heure)
    old, new = FindLocation(first, heure, duree, nom, jour, bool)
    UpdateInside(old, new)

def getList():
    return Disponible["Lundi"]
#nouvelle_tache(heure, duree, nom)
