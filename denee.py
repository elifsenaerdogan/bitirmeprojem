# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:55:38 2020

@author: user
"""

import pandas as pd
import numpy as np




veriler=pd.read_csv('projeverisetim123.csv')
print(veriler)


from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

having_ip_adress=veriler.iloc[:,0:1].values
print(having_ip_adress)


from sklearn import preprocessing
le=preprocessing.LabelEncoder()

having_ip_adress[:,0]=le.fit_transform(veriler.iloc[:,0])
print(having_ip_adress)

'''
ohe=preprocessing.OneHotEncoder()
having_ip_adress=ohe.fit_transform(having_ip_adress).toarray()
print(having_ip_adress)
'''


from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

shortning_service=veriler.iloc[:,2:3].values
'''
print(shortning_service)

'''
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

shortning_service[:,0]=le.fit_transform(veriler.iloc[:,2:3])
print(shortning_service)

'''
ohe=preprocessing.OneHotEncoder()
shortning_service=ohe.fit_transform(shortning_service).toarray()
print(shortning_service)
'''

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

having_at_symbol=veriler.iloc[:,3:4].values
print(having_at_symbol)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

having_at_symbol[:,0]=le.fit_transform(veriler.iloc[:,3:4])
print(having_at_symbol)
'''
ohe=preprocessing.OneHotEncoder()
having_at_symbol=ohe.fit_transform(having_at_symbol).toarray()
print(having_at_symbol)
'''
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

prefix_suffix=veriler.iloc[:,5:6].values
print(prefix_suffix)

le=preprocessing.LabelEncoder()
prefix_suffix[:,0]=le.fit_transform(veriler.iloc[:,5:6])
print(prefix_suffix)


from sklearn import preprocessing
'''
ohe=preprocessing.OneHotEncoder()
prefix_suffix=ohe.fit_transform(prefix_suffix).toarray()
print(prefix_suffix)
'''
sslfinal_state=veriler.iloc[:,7:8].values
print(sslfinal_state)
'''
ohe=preprocessing.OneHotEncoder()
sslfinal_state=ohe.fit_transform(sslfinal_state).toarray()
print(sslfinal_state)
'''
le=preprocessing.LabelEncoder()
sslfinal_state[:,0]=le.fit_transform(veriler.iloc[:,7:8])


favicon=veriler.iloc[:,9:10].values
print(favicon)

le=preprocessing.LabelEncoder()
favicon[:,0]=le.fit_transform(veriler.iloc[:,9:10])
'''
ohe=preprocessing.OneHotEncoder()
favicon=ohe.fit_transform(favicon).toarray()
print(favicon)
'''
port=veriler.iloc[:,10:11].values
print(port)

le=preprocessing.LabelEncoder()
port[:,0]=le.fit_transform(veriler.iloc[:,10:11])
'''
ohe=preprocessing.OneHotEncoder()
port=ohe.fit_transform(port).toarray()
print(port)

'''
https_token=veriler.iloc[:,11:12].values
print(https_token)

le=preprocessing.LabelEncoder()
https_token[:,0]=le.fit_transform(veriler.iloc[:,11:12])
'''
ohe=preprocessing.OneHotEncoder()
https_token=ohe.fit_transform(https_token).toarray()
print(https_token)

'''
sfh=veriler.iloc[:,15:16].values
print(sfh)

le=preprocessing.LabelEncoder()
sfh[:,0]=le.fit_transform(veriler.iloc[:,15:16])

'''
ohe=preprocessing.OneHotEncoder()
sfh=ohe.fit_transform(sfh).toarray()
print(sfh)
'''
submitting_to_email=veriler.iloc[:,16:17].values
print(submitting_to_email)

le=preprocessing.LabelEncoder()
submitting_to_email[:,0]=le.fit_transform(veriler.iloc[:,16:17])
'''
ohe=preprocessing.OneHotEncoder()
submitting_to_email=ohe.fit_transform(submitting_to_email).toarray()
print(submitting_to_email)

'''
abnormal_url=veriler.iloc[:,17:18].values
print(abnormal_url)

le=preprocessing.LabelEncoder()
abnormal_url[:,0]=le.fit_transform(veriler.iloc[:,17:18])

on_mouse_over=veriler.iloc[:,19:20].values
print(on_mouse_over)

le=preprocessing.LabelEncoder()
on_mouse_over[:,0]=le.fit_transform(veriler.iloc[:,19:20])

rightclick=veriler.iloc[:,20:21].values
print(rightclick)

le=preprocessing.LabelEncoder()
rightclick[:,0]=le.fit_transform(veriler.iloc[:,20:21])

popupwindow=veriler.iloc[:,21:22].values
print(popupwindow)

le=preprocessing.LabelEncoder()
popupwindow[:,0]=le.fit_transform(veriler.iloc[:,21:22])

Iframe=veriler.iloc[:,22:23].values
print(Iframe)

le=preprocessing.LabelEncoder()
Iframe[:,0]=le.fit_transform(veriler.iloc[:,22:23])

DNSRecord=veriler.iloc[:,24:25].values
print(DNSRecord)

le=preprocessing.LabelEncoder()
DNSRecord[:,0]=le.fit_transform(veriler.iloc[:,24:25])

gogle_index=veriler.iloc[:,27:28].values
print(gogle_index)

le=preprocessing.LabelEncoder()
gogle_index[:,0]=le.fit_transform(veriler.iloc[:,27:28])

result=veriler.iloc[:,29:30].values
print(result)

le=preprocessing.LabelEncoder()
result[:,0]=le.fit_transform(veriler.iloc[:,29:30])

##birleştirme
print(list(range(20)))

sonuc1= pd.DataFrame(data=having_ip_adress, index = range(20), columns = ['having_ip_adress'])
print(sonuc1)

url_length=veriler.iloc[:,1:2].values
sonuc2=pd.DataFrame(data=url_length,index = range(20), columns = ['url_length'])
print(sonuc2)

sonuc3=pd.DataFrame(data=shortning_service, index = range(20), columns = ['shortning_service'])
print(sonuc3)

sonuc4=pd.DataFrame(data=having_at_symbol, index = range(20), columns = ['having_at_symbol'])
print(sonuc4)

double_slash_redirecting=veriler.iloc[:,4:5].values
sonuc5=pd.DataFrame(data=double_slash_redirecting, index = range(20), columns = ['double_slash_redirecting'])
print(sonuc5)

sonuc6= pd.DataFrame(data=prefix_suffix, index = range(20), columns = ['prefix_suffix'])
print(sonuc6)

having_sub_domain=veriler.iloc[:,6:7]
sonuc7= pd.DataFrame(data=having_sub_domain, index = range(20), columns = ['having_sub_domain'])
print(sonuc7)

sonuc8= pd.DataFrame(data=sslfinal_state, index = range(20), columns = ['sslfinal_state'])
print(sonuc8)

domain_registerion_length=veriler.iloc[:,8:9].values
sonuc9= pd.DataFrame(data=domain_registerion_length, index = range(20), columns = ['domain_registerion_length'])
print(sonuc9)

sonuc10= pd.DataFrame(data=favicon, index = range(20), columns = ['favicon'])
print(sonuc10)

sonuc11= pd.DataFrame(data=port, index = range(20), columns = ['port'])
print(sonuc11)

sonuc12= pd.DataFrame(data=https_token, index = range(20), columns = ['https_token'])
print(sonuc12)

request_url=veriler.iloc[:,12:13].values
sonuc13= pd.DataFrame(data=request_url, index = range(20), columns = ['request_url'])
print(sonuc13)

url_of_anchor=veriler.iloc[:,13:14].values
sonuc14= pd.DataFrame(data=url_of_anchor, index = range(20), columns = ['url_of_anchor'])
print(sonuc14)

links_in_tags=veriler.iloc[:,14:15].values
sonuc15= pd.DataFrame(data=links_in_tags, index = range(20), columns = ['links_in_tags'])
print(sonuc15)

sonuc16= pd.DataFrame(data=sfh, index = range(20), columns = ['sfh'])
print(sonuc16)

sonuc17= pd.DataFrame(data=submitting_to_email, index = range(20), columns = ['submitting_to_email'])
print(sonuc17)

sonuc18= pd.DataFrame(data=abnormal_url, index = range(20), columns = ['abnormal_url'])
print(sonuc18)

redirect=veriler.iloc[:,18:19].values
sonuc19= pd.DataFrame(data=redirect, index = range(20), columns = ['redirect'])
print(sonuc19)

sonuc20= pd.DataFrame(data=on_mouse_over, index = range(20), columns = ['on_mouse_over'])
print(sonuc20)

sonuc21= pd.DataFrame(data=rightclick, index = range(20), columns = ['rightclick'])
print(sonuc21)

sonuc22= pd.DataFrame(data=popupwindow, index = range(20), columns = ['popupwindow'])
print(sonuc22)

sonuc23= pd.DataFrame(data=Iframe, index =range(20), columns = ['Iframe'])
print(sonuc23)

age_of_domain=veriler.iloc[:,23:24].values
sonuc24= pd.DataFrame(data=age_of_domain, index =range(20), columns = ['age_of_domain'])
print(sonuc24)

sonuc25= pd.DataFrame(data=DNSRecord, index =range(20), columns = ['DNSRecord'])
print(sonuc25)

web_traffic=veriler.iloc[:,25:26].values
sonuc26= pd.DataFrame(data=web_traffic, index =range(20), columns = ['web_traffic'])
print(sonuc26)

page_rank=veriler.iloc[:,26:27].values
sonuc27= pd.DataFrame(data=page_rank, index =range(20), columns = ['page_rank'])
print(sonuc27)

sonuc28= pd.DataFrame(data=gogle_index, index =range(20), columns = ['gogle_index'])
print(sonuc28)

links_pointing_to_page=veriler.iloc[:,28:29].values
sonuc29= pd.DataFrame(data=links_pointing_to_page, index =range(20), columns = ['links_pointing_to_page'])
print(sonuc29)

sonuc30= pd.DataFrame(data=result, index =range(20), columns = ['result'])
print(sonuc30)

##tablo birleştirme


s=pd.concat([sonuc1,sonuc2,sonuc3,sonuc4,sonuc5,sonuc6,sonuc7,sonuc8,sonuc9,sonuc10,sonuc11,sonuc12,sonuc13,sonuc14,sonuc15,sonuc16,sonuc17,sonuc18,sonuc19,sonuc20,sonuc21,sonuc22,sonuc23,sonuc24,sonuc25,sonuc26,sonuc27,sonuc28,sonuc29],axis=1)
print(s)

s2=pd.concat([sonuc30],axis=1)
print(s2)

s3=pd.concat([s,s2],axis=1)
print(s3)



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,s2,test_size=0.33, random_state=0)

#verilerin olceklenmesi


##standardizasyon
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

'''
#normalizasyon

from sklearn.preprocessing import MinMaxScaler

mmc=MinMaxScaler()

X_train = mmc.fit_transform(x_train)
X_test = mmc.transform(x_test)

'''


#öğretme
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)

#tahminettirme
y_pred=logreg.predict(X_test)
print(y_pred)
y_pred=pd.DataFrame(data=y_pred)



'''
#numpy dizi dönüşümü
y=y_test.values
yhat=y_pred.values



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predk)
print(cm)


#kenyakınsınıflandırma

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)
y_predk=knn.predict(X_test)
print('kenyakin')
print(y_predk)
print(y_test)

'''
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_predk=classifier.predict(X_test)
print('kenyakin')
print(y_predk)
print(y_test)
'''



y_predk=pd.DataFrame(data=y_predk)



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predk)
print(cm)

'''
#destekvektörregresyonusınıflandırma(svc)

from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_predsvc=svc.predict(X_test)
print('svc')
print(y_predsvc)

y_predsvc=pd.DataFrame(data=y_predsvc)


'''
#naivebayes

from sklearn.naive_bayes import MultinominalNB
mnb=MultinominalNB()
mnb.fit(X_train,y_train)
y_prednb=mnb.predict(X_test)

'''
#kararargacıilesınıflandırma(decisiontree)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_predka=dtc.predict(X_test)

print('kararagaci')
print(y_predka)


y_predka=pd.DataFrame(data=y_predka)




#rassalormansınıflandırma(randomforest)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)
y_predro=rfc.predict(X_test)
print('rassal')
print(y_predro)


y_predro=pd.DataFrame(data=y_predro)


veriler.info()

veriler['having_sub_domain']=veriler['having_sub_domain'].astype('int')



'''
import sklearn.metrics as metrics
probs=rfc.predict_proba(X_test)
preds=probs[:,1]
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)



#ROC TPR,FPR DEGERLERİ
y_tah=rfc.predict_proba(X_test)
print(y_tah[:,0])

from sklearn import metrics
fpr,tpr,thold=metrics.roc_curve(y_test, y_tah[:,0], pos_label='m')

print(fpr)
print(tpr)


'''
print(y_test)


'''
from tkinter import *
from tkcalendar import DateEntry


 
master = Tk()

canvas=Canvas(master,height=450,width=750)
canvas.pack()

frame_ust=Frame(master,bg='#add8e6')
frame_ust.place(relx=0.1,rely=0.1,relwidth=0.8,relheight=0.1)

frame_alt_sol=Frame(master,bg='#add8e6')
frame_alt_sol.place(relx=0.1,rely=0.23,relwidth=0.23,relheight=0.5)

frame_alt_sag=Frame(master,bg='#add8e6')
frame_alt_sag.place(relx=0.35,rely=0.23,relwidth=0.55,relheight=0.5)

hatirlatma_tipi_etiket=Label(frame_ust,bg='#add8e6',text="Hatırlatma Tipi:",font="Verdana 10 bold")
hatirlatma_tipi_etiket.pack(padx=10,pady=10,side=LEFT)

hatirlatma_tipi_opsiyon=StringVar(frame_ust)
hatirlatma_tipi_opsiyon.set("\t")
hatirlatma_tipi_acilir_menu=OptionMenu(frame_ust, hatirlatma_tipi_opsiyon,"Dogum günü","Fatura","Yıldönümü")
hatirlatma_tipi_acilir_menu.pack(padx=10,pady=10,side=LEFT)

hatirlatma_tarihi_secici=DateEntry(frame_ust,width=12,background='orange',foreground='black',borderwidth=1,locale="de_De")
hatirlatma_tarihi_secici._top_cal.overrideredirect(False)
hatirlatma_tarihi_secici.pack(padx=10,pady=10,side=RIGHT)

hatirlatma_tipi_etiket=Label(frame_ust,bg='#add8e6',text="Hatırlatma Tarihi:",font="Verdana 10 bold")
hatirlatma_tipi_etiket.pack(padx=10,pady=10,side=RIGHT)


Label(frame_alt_sol,bg='#add8e6',text="Hatırlatma Yöntemi",font="Verdana 10 bold").pack(padx=10,pady=10,anchor=NW)

var=IntVar()
R1=Radiobutton(frame_alt_sol,text="Sisteme Kaydet",variable=var,value=1,bg='#add8e6',font="Verdana 10 bold")
R1.pack(anchor=NW,padx=5,pady=10)

R2=Radiobutton(frame_alt_sol,text="E-posta gönder",variable=var,value=2,bg='#add8e6',font="Verdana 10 bold")
R2.pack(anchor=NW,padx=5,pady=10)

var1=IntVar()
C1=Checkbutton(frame_alt_sol,text="Bir ay önce",variable=var1,onvalue=1,offvalue=0,bg='#add8e6',font="Verdana 10 bold")
C1.pack(anchor=NW,pady=2,padx=25)

var2=IntVar()
C2=Checkbutton(frame_alt_sol,text="Bir günönce",variable=var2,onvalue=1,offvalue=0,bg='#add8e6',font="Verdana 10 bold")
C2.pack(anchor=NW,pady=2,padx=25)

var3=IntVar()
C3=Checkbutton(frame_alt_sol,text="Aynı güne",variable=var3,onvalue=1,offvalue=0,bg='#add8e6',font="Verdana 10 bold")
C3.pack(anchor=NW,pady=2,padx=25)

from tkinter import messagebox

def gonder():
   
    son_mesaj = ""
    
    try:

     if var.get():
          if var.get() ==  1:
               son_mesaj += "Veriniz başarıyla sisteme kaydedildi."
               
               tip=hatirlatma_tipi_opsiyon.get() if hatirlatma_tipi_opsiyon.get() == ' ' else "Genel"
               
               tarih=hatirlatma_tarihi_secici.get()
               
               mesaj=metin_alani.get("1.0","end")
               
               with open("hatirlatmalar.txt","w") as dosya:
                   dosya.write('{} kategorinde,{} tarihine ve "{}" notuyla hatirlatma'.format(tip,tarih,mesaj))
                   
                   dosya.close()
                               
                               
               
          elif var.get() == 2:
               son_mesaj += "E-Posta gönderilecek."
          messagebox.showinfo("Başarılı işlem",son_mesaj)
               
     else:
          son_mesaj += "Gerekli alanların dolduruldugundan emin olunuz!"
          messagebox.showwarning("yetersiz bilgi",son_mesaj)
          
    except:
         son_mesaj += "İşlem başarısız!"
         messagebox.showerror("başarısız işlem",son_mesaj)
          
               
               
    finally:
        master.destroy()


Label(frame_alt_sag,bg='#add8e6',text="Hatırlatma Mesaji",font="Verdana 10 bold").pack(padx=10,pady=10,anchor=NW)

metin_alani=Text(frame_alt_sag,height=9,width=50)
metin_alani.tag_configure('style',foreground='#bfbfbf',font=('Verdana',7,'bold'))
metin_alani.pack()
karsilama_metni='Mesajınızı buraya giriniz'

metin_alani.insert(END, karsilama_metni,'style')

gonder_butonu=Button(frame_alt_sag,text="Gönder",command=gonder)

gonder_butonu.pack(anchor=S)

    

master.mainloop()
 
'''
import tkinter as tk

from tkinter import Label
from tkinter import LEFT
from tkinter import Frame
from tkinter import Canvas
from tkinter import Text
from tkinter import END
from tkinter import IntVar
from tkinter import Radiobutton
from tkinter import RIGHT
from tkinter import Toplevel
from tkinter import Button




   



master = tk.Tk()

canvas=Canvas(master,height=450,width=750)
canvas.pack()

frame_ust=Frame(master,bg='#add8e6')
frame_ust.place(relx=0.1,rely=0.1,relwidth=0.8,relheight=0.1)


Label(frame_ust,bg='#add8e6',text="Web Sitesinin İsmi:",font="Verdana 10 bold").pack(padx=10,pady=10,side=LEFT)



isim_alani=Text(frame_ust,height=4,width=40)
isim_alani.tag_configure('style',foreground='#bfbfbf',font=('Verdana',8,'bold'))
isim_alani.pack(side=LEFT)
karsilama_metni='Sitenin ismini buraya giriniz'

isim_alani.insert(END, karsilama_metni,'style')

'''
dosya=open("websiteismi.txt","w")
print(isim_alani,file=dosya,end='.')
dosya.close()
'''

frame_alt=Frame(master,bg='#add8e6')
frame_alt.place(relx=0.1,rely=0.23,relheight=0.1,relwidth=0.8)

Label(frame_alt,bg='#add8e6',text="Web Sitesinin Özellikleri",font="Verdana 10 bold").pack(padx=10,pady=10)

frame_alt_sol_1=Frame(master,bg='#add8e6')
frame_alt_sol_1.place(relx=0.1,rely=0.34,relheight=0.1,relwidth=0.4)

Label(frame_alt_sol_1,bg='#add8e6',text="Iframe",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R1=Radiobutton(frame_alt_sol_1,text="0",variable=var,value=1,bg='#add8e6',font="Verdana 7 bold")
R1.pack(side=LEFT)
R2=Radiobutton(frame_alt_sol_1,text="1",variable=var,value=2,bg='#add8e6',font="Verdana 7 bold")
R2.pack(side=RIGHT)


frame_alt_sol_2=Frame(master,bg='#add8e6')
frame_alt_sol_2.place(relx=0.1,rely=0.45,relheight=0.1,relwidth=0.4)

Label(frame_alt_sol_2,bg='#add8e6',text="having_ip_address",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R3=Radiobutton(frame_alt_sol_2,text="0",variable=var,value=3,bg='#add8e6',font="Verdana 7 bold")
R3.pack(side=LEFT)
R4=Radiobutton(frame_alt_sol_2,text="1",variable=var,value=4,bg='#add8e6',font="Verdana 7 bold")
R4.pack(side=RIGHT)

frame_alt_sol_3=Frame(master,bg='#add8e6')
frame_alt_sol_3.place(relx=0.1,rely=0.56,relheight=0.1,relwidth=0.4)

Label(frame_alt_sol_3,bg='#add8e6',text="shortning_service",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R5=Radiobutton(frame_alt_sol_3,text="0",variable=var,value=5,bg='#add8e6',font="Verdana 7 bold")
R5.pack(side=LEFT)
R6=Radiobutton(frame_alt_sol_3,text="1",variable=var,value=6,bg='#add8e6',font="Verdana 7 bold")
R6.pack(side=RIGHT)



frame_alt_sol_4=Frame(master,bg='#add8e6')
frame_alt_sol_4.place(relx=0.1,rely=0.67,relheight=0.1,relwidth=0.4)


Label(frame_alt_sol_4,bg='#add8e6',text="having_at_symbol",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R7=Radiobutton(frame_alt_sol_4,text="0",variable=var,value=7,bg='#add8e6',font="Verdana 7 bold")
R7.pack(side=LEFT)
R8=Radiobutton(frame_alt_sol_4,text="1",variable=var,value=8,bg='#add8e6',font="Verdana 7 bold")
R8.pack(side=RIGHT)

frame_alt_sol_5=Frame(master,bg='#add8e6')
frame_alt_sol_5.place(relx=0.1,rely=0.78,relheight=0.1,relwidth=0.4)

Label(frame_alt_sol_5,bg='#add8e6',text="prefix_suffix",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R9=Radiobutton(frame_alt_sol_5,text="0",variable=var,value=9,bg='#add8e6',font="Verdana 7 bold")
R9.pack(side=LEFT)
R10=Radiobutton(frame_alt_sol_5,text="1",variable=var,value=10,bg='#add8e6',font="Verdana 7 bold")
R10.pack(side=RIGHT)

frame_alt_sag_1=Frame(master,bg='#add8e6')
frame_alt_sag_1.place(relx=0.51,rely=0.34,relheight=0.1,relwidth=0.4)

Label(frame_alt_sag_1,bg='#add8e6',text="sslfinal_state",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R11=Radiobutton(frame_alt_sag_1,text="0",variable=var,value=11,bg='#add8e6',font="Verdana 7 bold")
R11.pack(side=LEFT)
R12=Radiobutton(frame_alt_sag_1,text="1",variable=var,value=12,bg='#add8e6',font="Verdana 7 bold")
R12.pack(side=LEFT)
R13=Radiobutton(frame_alt_sag_1,text="2",variable=var,value=13,bg='#add8e6',font="Verdana 7 bold")
R13.pack(side=RIGHT)

frame_alt_sag_2=Frame(master,bg='#add8e6')
frame_alt_sag_2.place(relx=0.51,rely=0.45,relheight=0.1,relwidth=0.4)

Label(frame_alt_sag_2,bg='#add8e6',text="favicon",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R14=Radiobutton(frame_alt_sag_2,text="0",variable=var,value=14,bg='#add8e6',font="Verdana 7 bold")
R14.pack(side=LEFT)
R15=Radiobutton(frame_alt_sag_2,text="1",variable=var,value=15,bg='#add8e6',font="Verdana 7 bold")
R15.pack(side=RIGHT)

frame_alt_sag_3=Frame(master,bg='#add8e6')
frame_alt_sag_3.place(relx=0.51,rely=0.56,relheight=0.1,relwidth=0.4)

Label(frame_alt_sag_3,bg='#add8e6',text="port",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R16=Radiobutton(frame_alt_sag_3,text="0",variable=var,value=16,bg='#add8e6',font="Verdana 7 bold")
R16.pack(side=LEFT)
R17=Radiobutton(frame_alt_sag_3,text="1",variable=var,value=17,bg='#add8e6',font="Verdana 7 bold")
R17.pack(side=RIGHT)

frame_alt_sag_4=Frame(master,bg='#add8e6')
frame_alt_sag_4.place(relx=0.51,rely=0.67,relheight=0.1,relwidth=0.4)

Label(frame_alt_sag_4,bg='#add8e6',text="https_token",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R18=Radiobutton(frame_alt_sag_4,text="0",variable=var,value=18,bg='#add8e6',font="Verdana 7 bold")
R18.pack(side=LEFT)
R19=Radiobutton(frame_alt_sag_4,text="1",variable=var,value=19,bg='#add8e6',font="Verdana 7 bold")
R19.pack(side=RIGHT)

frame_alt_sag_5=Frame(master,bg='#add8e6')
frame_alt_sag_5.place(relx=0.51,rely=0.78,relheight=0.1,relwidth=0.4)

Label(frame_alt_sag_5,bg='#add8e6',text="sfh",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)

var=IntVar()
R20=Radiobutton(frame_alt_sag_5,text="0",variable=var,value=20,bg='#add8e6',font="Verdana 7 bold")
R20.pack(side=LEFT)
R21=Radiobutton(frame_alt_sag_5,text="1",variable=var,value=21,bg='#add8e6',font="Verdana 7 bold")
R21.pack(side=RIGHT)

##toplevel ile pencere ekleme
def pencere_olustur():
   
    pencere=Toplevel(bg="white")
    
    canvas=Canvas(pencere,height=450,width=750)
    canvas.pack()
    
    frame_sol_1=Frame(pencere,bg='#add8e6')
    frame_sol_1.place(relx=0.1,rely=0.1,relheight=0.1,relwidth=0.4)

    Label(frame_sol_1,bg='#add8e6',text="submitting_to_email",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R22=Radiobutton(frame_sol_1,text="0",variable=var,value=22,bg='#add8e6',font="Verdana 7 bold")
    R22.pack(side=LEFT)
    R23=Radiobutton(frame_sol_1,text="1",variable=var,value=23,bg='#add8e6',font="Verdana 7 bold")
    R23.pack(side=RIGHT)
    
    frame_sol_2=Frame(pencere,bg='#add8e6')
    frame_sol_2.place(relx=0.1,rely=0.22,relheight=0.1,relwidth=0.4)

    Label(frame_sol_2,bg='#add8e6',text="abnormal_url",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R24=Radiobutton(frame_sol_2,text="0",variable=var,value=24,bg='#add8e6',font="Verdana 7 bold")
    R24.pack(side=LEFT)
    R25=Radiobutton(frame_sol_2,text="1",variable=var,value=25,bg='#add8e6',font="Verdana 7 bold")
    R25.pack(side=RIGHT)
    
    frame_sol_3=Frame(pencere,bg='#add8e6')
    frame_sol_3.place(relx=0.1,rely=0.34,relheight=0.1,relwidth=0.4)

    Label(frame_sol_3,bg='#add8e6',text="on_mouse_over",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R26=Radiobutton(frame_sol_3,text="0",variable=var,value=26,bg='#add8e6',font="Verdana 7 bold")
    R26.pack(side=LEFT)
    R27=Radiobutton(frame_sol_3,text="1",variable=var,value=27,bg='#add8e6',font="Verdana 7 bold")
    R27.pack(side=RIGHT)
    
    frame_sol_4=Frame(pencere,bg='#add8e6')
    frame_sol_4.place(relx=0.1,rely=0.45,relheight=0.1,relwidth=0.4)

    Label(frame_sol_4,bg='#add8e6',text="rightclick",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R28=Radiobutton(frame_sol_4,text="0",variable=var,value=28,bg='#add8e6',font="Verdana 7 bold")
    R28.pack(side=LEFT)
    R29=Radiobutton(frame_sol_4,text="1",variable=var,value=29,bg='#add8e6',font="Verdana 7 bold")
    R29.pack(side=RIGHT)
    
    frame_sol_5=Frame(pencere,bg='#add8e6')
    frame_sol_5.place(relx=0.1,rely=0.56,relheight=0.1,relwidth=0.4)

    Label(frame_sol_5,bg='#add8e6',text="popupwindow",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R30=Radiobutton(frame_sol_5,text="0",variable=var,value=30,bg='#add8e6',font="Verdana 7 bold")
    R30.pack(side=LEFT)
    R31=Radiobutton(frame_sol_5,text="1",variable=var,value=31,bg='#add8e6',font="Verdana 7 bold")
    R31.pack(side=RIGHT)
    
    frame_sol_6=Frame(pencere,bg='#add8e6')
    frame_sol_6.place(relx=0.1,rely=0.67,relheight=0.1,relwidth=0.4)

    Label(frame_sol_6,bg='#add8e6',text="DNSRecord",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R32=Radiobutton(frame_sol_6,text="0",variable=var,value=32,bg='#add8e6',font="Verdana 7 bold")
    R32.pack(side=LEFT)
    R33=Radiobutton(frame_sol_6,text="1",variable=var,value=33,bg='#add8e6',font="Verdana 7 bold")
    R33.pack(side=RIGHT)
    
    frame_sol_7=Frame(pencere,bg='#add8e6')
    frame_sol_7.place(relx=0.1,rely=0.78,relheight=0.1,relwidth=0.4)

    Label(frame_sol_7,bg='#add8e6',text="gogle_index",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    var=IntVar()
    R34=Radiobutton(frame_sol_7,text="0",variable=var,value=34,bg='#add8e6',font="Verdana 7 bold")
    R34.pack(side=LEFT)
    R35=Radiobutton(frame_sol_7,text="1",variable=var,value=35,bg='#add8e6',font="Verdana 7 bold")
    R35.pack(side=RIGHT)
    
    
    frame_sag_1=Frame(pencere,bg='#add8e6')
    frame_sag_1.place(relx=0.51,rely=0.1,relheight=0.1,relwidth=0.4)

    Label(frame_sag_1,bg='#add8e6',text="url_length",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_1=Text(frame_sag_1,height=4,width=20)
    giris_1.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_1.pack()
    
   
    
    frame_sag_2=Frame(pencere,bg='#add8e6')
    frame_sag_2.place(relx=0.51,rely=0.22,relheight=0.1,relwidth=0.4)



    Label(frame_sag_2,bg='#add8e6',text="double_slash_redirecting",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_2=Text(frame_sag_2,height=4,width=10)
    giris_2.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_2.pack()
    
    frame_sag_3=Frame(pencere,bg='#add8e6')
    frame_sag_3.place(relx=0.51,rely=0.34,relheight=0.1,relwidth=0.4)


    Label(frame_sag_3,bg='#add8e6',text="having_sub_domain",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_3=Text(frame_sag_3,height=4,width=15)
    giris_3.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_3.pack()
   
    
    
    frame_sag_4=Frame(pencere,bg='#add8e6')
    frame_sag_4.place(relx=0.51,rely=0.45,relheight=0.1,relwidth=0.4)


    Label(frame_sag_4,bg='#add8e6',text="domain_registerion_length",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_4=Text(frame_sag_4,height=4,width=10)
    giris_4.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_4.pack()
    
    
    
    frame_sag_5=Frame(pencere,bg='#add8e6')
    frame_sag_5.place(relx=0.51,rely=0.56,relheight=0.1,relwidth=0.4)


    Label(frame_sag_5,bg='#add8e6',text="request_url",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_5=Text(frame_sag_5,height=4,width=18)
    giris_5.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_5.pack()
   
    frame_sag_6=Frame(pencere,bg='#add8e6')
    frame_sag_6.place(relx=0.51,rely=0.67,relheight=0.1,relwidth=0.4)


    Label(frame_sag_6,bg='#add8e6',text="url_of_anchor",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_6=Text(frame_sag_6,height=4,width=18)
    giris_6.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_6.pack() 
    
    
    frame_sag_7=Frame(pencere,bg='#add8e6')
    frame_sag_7.place(relx=0.51,rely=0.78,relheight=0.1,relwidth=0.4)


    Label(frame_sag_7,bg='#add8e6',text="links_in_tags",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
    giris_7=Text(frame_sag_7,height=4,width=18)
    giris_7.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
    giris_7.pack() 
    
    b1=Button(pencere,text="Diğer sayfaya geçiniz.",command=yeni_pencere_olustur)
    b1.place(relx=0.70,rely=0.90)

       
    
def yeni_pencere_olustur(): 
      
      yeni_pencere = Toplevel(bg="white")
     
      canvas=Canvas(yeni_pencere,height=450,width=750)
      canvas.pack()
      
      frame_yeni_sol_1=Frame(yeni_pencere,bg='#add8e6')
      frame_yeni_sol_1.place(relx=0.1,rely=0.1,relheight=0.1,relwidth=0.4)
      
      Label(frame_yeni_sol_1,bg='#add8e6',text="redirect",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
      giris_8=Text(frame_yeni_sol_1,height=4,width=18)
      giris_8.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
      giris_8.pack() 
      
      
      frame_yeni_sol_2=Frame(yeni_pencere,bg='#add8e6')
      frame_yeni_sol_2.place(relx=0.1,rely=0.22,relheight=0.1,relwidth=0.4)
      
      Label(frame_yeni_sol_2,bg='#add8e6',text="age_of_domain",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
      giris_9=Text(frame_yeni_sol_2,height=4,width=18)
      giris_9.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
      giris_9.pack() 
      
      frame_yeni_sol_3=Frame(yeni_pencere,bg='#add8e6')
      frame_yeni_sol_3.place(relx=0.1,rely=0.34,relheight=0.1,relwidth=0.4)
      
      Label(frame_yeni_sol_3,bg='#add8e6',text="web_traffic",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
      giris_10=Text(frame_yeni_sol_3,height=4,width=18)
      giris_10.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
      giris_10.pack() 
      
      frame_yeni_sag_1=Frame(yeni_pencere,bg='#add8e6')
      frame_yeni_sag_1.place(relx=0.51,rely=0.1,relheight=0.1,relwidth=0.4)
      
      Label(frame_yeni_sag_1,bg='#add8e6',text="page_rank",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
      giris_11=Text(frame_yeni_sag_1,height=4,width=18)
      giris_11.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
      giris_11.pack() 
      
      frame_yeni_sag_2=Frame(yeni_pencere,bg='#add8e6')
      frame_yeni_sag_2.place(relx=0.51,rely=0.22,relheight=0.1,relwidth=0.4)
      
      Label(frame_yeni_sag_2,bg='#add8e6',text="links_in_pointing",font="Verdana 8 bold").pack(padx=10,pady=10,side=LEFT)
    
      giris_12=Text(frame_yeni_sag_2,height=4,width=18)
      giris_12.tag_configure('style',foreground='#bfbfbf',font=('Verdana',10,'bold'))
      giris_12.pack() 
    
   
    
    
    
    
    
    
    
    
    

   

b=Button(master,text="Diğer sayfaya geçiniz.",command=pencere_olustur)
b.place(relx=0.70,rely=0.90)



























master.mainloop()
