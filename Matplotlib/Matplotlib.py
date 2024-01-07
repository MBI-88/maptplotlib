#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt 
import numpy as np
import math


# ***Matplotlib_1***

# In[5]:


x = range(1000)
y = [i**2 for i in x]
plt.plot(x,y)
plt.show()


# In[6]:


T = range(200)
x = [(2*math.pi*t)/len(T) for t in T]
y = [math.sin(value) for value in x]
plt.plot(x,y)
plt.show()


# In[7]:


# Usando numpy
x = np.linspace(0,2*np.pi,200)
y = np.sin(x)
plt.plot(x,y)
plt.show()


# In[8]:


x = np.linspace(-2,6,100)
y = x**2 -3*x + 20
plt.plot(x,y)
plt.show()


# ***Matplotlib_2***

# In[9]:


# Multiples graficas

# comandos basicos
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.ioff() #para separar la grafica del notebook usando %matplotlib 

x = np.linspace(0,2*np.pi,200)
ya = np.sin(x)
yb = np.cos(x)
plt.plot(x,ya)
plt.plot(x,yb)
plt.show()
#plt.ion() igual plt.show()


# In[10]:


# Renderizacion de imagenes
def plotPendiente(X,Y):
    Xs = X[1:] -X[:-1]
    Ys = Y[1:] -Y[:-1]
    plt.plot(X[1:],Ys/Xs)

X = np.linspace(-3,3,100)
Y = np.exp(-X**2)
plt.plot(X,Y)
plotPendiente(X,Y)
    


# In[11]:


# Plotiando curvas de un archivo. Falsta el archivo
"""
X,Y = [],[]
for line in open('datos.txt','r'):
    valores = [float(s) for s in line.split(',')]
    X.append(valores[0])
    Y.append(valores[1])
plt.plot(X,Y)
plt.show()
"""


# In[12]:


data = np.loadtxt('datos.txt',delimiter=',') # Solo datos numericos
data


# ***Matplotlib_3***

# In[13]:


# Graficando Barras
data = np.random.rand(1024,2)
plt.scatter(data[:,0],data[:,1])
plt.show()


# In[14]:


data = [5.,25.,50.,20.]
plt.bar(range(len(data)),data)
plt.show()


# In[15]:


plt.bar([0,1,4,9],data)
plt.show()


# In[16]:


# Cambiando el grosor de la barra
plt.bar([0,1,4,9],data,width=0.5)
plt.show()


# In[17]:


# Barras horizontales
plt.barh(range(len(data)),data)
plt.show()


# In[18]:


data = [
    [5.,25.,50.,20.],
    [4.,23.,51.,17.],
    [6.,22.,52.,19.]
]
X=np.arange(4)
plt.bar(X+0.0,data[0],color='b',width=0.25)
plt.bar(X+0.25,data[1],color='r',width=0.25)
plt.bar(X+0.50,data[2],color='g',width=0.25)
plt.show()


# In[19]:


lista_colores =['r','b','g']
brecha = 0.9/len(data)
for i,fila in enumerate(data):
    X = np.arange(len(fila))
    plt.bar(X+ i*brecha,fila,width=brecha,color=lista_colores[i%len(lista_colores)])


# In[20]:


# Graficando barras verticales apiladas
A = [5.,30.,45.,22.]
B = [5.,25.,50.,20.]
X=np.arange(4)
plt.bar(X,A,color='b',)
plt.bar(X,B,color='g',bottom=A)
plt.show()


# ***Matplotlib_4***

# In[21]:


# Graficas apiladas

data = np.array([
    [5.,30.,45.,22.],
    [5.,25.,50.,20.,],
    [1.,2.,1.,1.],
])
lista_de_colores = ['b','r','g']
X = np.arange(data.shape[1])
for i in range(data.shape[0]):
    plt.bar(X,data[i],bottom = np.sum(data[:i],axis=0),color = lista_de_colores[i % len(lista_de_colores)])

plt.show()


# In[22]:


# Barras de espaldas con espaldas
pob_mujeres = np.array([5.,30.,45.,22.])
pob_hombres = np.array([5.,25.,50.,20.])
X = np.arange(4)
plt.barh(X,pob_mujeres,color= 'g')
plt.barh(X,-pob_hombres,color = 'r')
plt.show()


# In[23]:


# Graficos de pastel

data = [5.,25.,50.,20.]
plt.pie(data)
plt.show()


# In[24]:


# Histogramas
X = np.random.randn(1000)
plt.hist(X,bins=20,color='g')
plt.show()


# In[25]:


# Graficos de caja o vigote

plt.boxplot(data)
plt.show()


# In[26]:


# Graficas de triangulaciones
import matplotlib.tri as tri


data = np.random.rand(100,200)
triangulos = tri.Triangulation(data[:,0],data[:,1])
plt.triplot(triangulos)
plt.show()


# ***Matplotlib_5***

# In[27]:


# Personalizando graficas

def pdf(X,mu,sigma):
    a=1./(sigma*np.sqrt(2.*np.pi))
    b=-1./(2.*sigma**2)
    return a*np.exp(b*(X-mu)**2)
X = np.linspace(-6,6,1000)
for i in range(5):
    sample=np.random.standard_normal(50)
    mu,sigma=np.mean(sample),np.std(sample)
    plt.plot(X,pdf(X,mu,sigma),color=(1,0.0,0.0,0.5))# Colores en RGB mas uno (A) que es transparencia
plt.plot(X,pdf(X,0.,1.),color='b')
plt.show()


# In[28]:


# Colores en scatter

A=np.random.standard_normal((100,2))
B=np.random.standard_normal((100,2))
plt.scatter(A[:,0],A[:,1],color='.25')
plt.scatter(B[:,0],B[:,1],color='y')
plt.show()


# In[29]:


# Bordes en las graficas

data= np.random.standard_normal((100,2))
plt.scatter(data[:,0],data[:,1],color='g',edgecolor='r') # Edgecolor para dar borde a la grafica
plt.show()


# ***Matplotlib_6***

# In[30]:


valores=np.random.randint(99,size=50)
set_colores=('r','g','.50','#C4B693')
lista_colores=[set_colores[(len(set_colores)*val) //100] for val in valores]
plt.bar(np.arange(len(valores)),valores,color=lista_colores)
plt.show()


# In[31]:


# Diagramas de pastel
valores=np.random.rand(8)
plt.pie(valores,colors=set_colores,shadow=True)
plt.show()


# In[32]:


# Diagrama de caja y vigote
valores=np.random.randn(100)
b=plt.boxplot(valores)
for nombres, lista_lineas in b.items():
    for linea in lista_lineas:
        linea.set_color('k')
plt.show()
        


# In[33]:


# Personalizando la grafica de caja y vigote
b=plt.boxplot(valores)
for nombres, lista_lineas in b.items():
    print(nombres)
    for linea in lista_lineas:
        if nombres == 'whiskers':linea.set_color('g')
        if nombres == 'caps': linea.set_color('r')
        if nombres == 'boxes': linea.set_color('b')
plt.show()


# In[34]:


# Graficos para scatter generando una espiral de colores
import matplotlib.cm as cm

N= 256
angulo=np.linspace(0,8*2*np.pi,N)
radio=np.linspace(.5,1.,N)
X= radio*np.cos(angulo) # Representacion polar
Y= radio*np.sin(angulo)
plt.scatter(X,Y,c=angulo,cmap=cm.afmhot,edgecolors='b')
plt.show()


# In[35]:


# Usando colormaps en graficos de barra
import matplotlib.colors as col
valores=np.random.randint(99,size=50) # 50 valores entre 1-99
cmap=cm.ScalarMappable(col.Normalize(0,99),cm.Accent)
plt.bar(np.arange(len(valores)),valores,color=cmap.to_rgba(valores),edgecolor='b')
plt.show()


# In[36]:


# Coloreando el patron de lineas y el grosor de linea

def pdf(X,mu,sigma):
    a=1./(sigma*np.sqrt(2.*np.pi))
    b=-1./(2.*sigma**2)
    return a*np.exp(b*(X-mu)**2)
X = np.linspace(-6,6,1024)
plt.plot(X,pdf(X,0.,1.),color='k',linestyle='solid')
plt.plot(X,pdf(X,0.,.5),color='k',linestyle='dashed')
plt.plot(X,pdf(X,0.,.25),color='k',linestyle='dashdot')
plt.show()


# ***Matplotlib_7***

# In[37]:


# Controlando el patron y el grosor de lineas
N=8
A=np.random.random(N)
B=np.random.random(N)
X=np.arange(N)
plt.barh(X,A,color='.3')
plt.barh(X,-B,color='0.75',ls='--',edgecolor='r')# cambiando ls
plt.show()


# In[38]:


# Ancho de linea

for i in range(64):
    muestras=np.random.standard_normal(50)
    mu,sigma=np.mean(muestras),np.std(muestras)
    plt.plot(X,pdf(X,mu,sigma),color='0.75',lw=.9)
plt.plot(X,pdf(X,0.,1.),color='g',lw=10.)# Moviendo lw
plt.show()


# In[39]:


# Patrones de relleno usndo el parametro hatch

plt.bar(X,A,color='g',hatch='+',ls='--',edgecolor='k')
plt.bar(X,A+B,bottom=A,color='w',hatch='/',edgecolor='k',ls=':')
plt.show()


# In[40]:


# Marcadores
A=np.random.standard_normal((100,2))
A += np.array((-1,-1))
B=np.random.standard_normal((100,2))
B += np.array((1,1))
plt.scatter(A[:,0],A[:,1],color='r',marker='v')# Cambiando el marker
plt.scatter(B[:,0],B[:,1],color='k',marker='p')
plt.show()


# In[41]:


# Udando el parametro markevery

X=np.linspace(-6,6,1024)
Y1=np.sinc(X)
Y2=np.sinc(X)+1
plt.plot(X,Y1,marker='o',color='.75')
plt.plot(X,Y2,marker='*',color='k',markevery=100)
plt.show()


# In[42]:


# Tamaño del marcador
A=np.random.standard_normal((100,2))
A += np.array((-1,-1))
B=np.random.standard_normal((100,2))
B += np.array((1,1))
plt.scatter(A[:,0],A[:,1],color='g',s=200,marker='o')# Cambiando el marker
plt.scatter(B[:,0],B[:,1],color='b',s=100,marker='o',edgecolor='k')
plt.show()


# In[43]:


M=np.random.standard_normal((1000,2))
R=np.sum(M**2,axis=1)
plt.scatter(M[:,0],M[:,1],c='w',marker='s',s=32.*R,edgecolor='k')
plt.show()


# In[44]:


# Creando tus propios marcadores

import matplotlib.path as mpath
from matplotlib import pyplot as plt
import matplotlib.patches as patches

shape_description=[
    (1.,2.,mpath.Path.MOVETO),
    (1.,1.,mpath.Path.LINETO),
    (2.,1.,mpath.Path.LINETO),
    (2.,-1.,mpath.Path.LINETO),
    (1.,-1.,mpath.Path.LINETO),
    (1.,-2.,mpath.Path.LINETO),
    (-1.,-2.,mpath.Path.LINETO),
    (-1.,-1.,mpath.Path.LINETO),
    (-2.,-1.,mpath.Path.LINETO),
    (-2.,1.,mpath.Path.LINETO),
    (-1.,1.,mpath.Path.LINETO),
    (-1.,2.,mpath.Path.LINETO),
    (0.,0.,mpath.Path.CLOSEPOLY),
]
u,v,codes=zip(*shape_description)
fig,ax=plt.subplots()
my_marker=mpath.Path(np.asarray((u,v)).T,codes)
patch=patches.PathPatch(my_marker,facecolor='orange',lw=5)
ax.add_patch(patch)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
plt.show()


# In[45]:


A=np.random.standard_normal((100,2))
A += np.array((-1,-1))
B=np.random.standard_normal((100,2))
B += np.array((1,1))
plt.scatter(A[:,0],A[:,1],color='y',marker=my_marker)# Cambiando el marker
plt.scatter(B[:,0],B[:,1],color='k',marker='p')
plt.show()


# In[46]:


X=np.linspace(-6,6,1024)
Y=np.sinc(X)
plt.plot(
    X,Y,
    lw=3.,
    color='k',
    markersize=9,
    markeredgewidth=2.5, # ancho del marcador
    markerfacecolor='.85',# color de relleno
    markeredgecolor='g',# color del borde
    marker=my_marker,# tipo de marca en este caso usando el personalizado
    markevery=32 # cada cuantos puntos se inserta la marca
)
plt.show()


# ***Matplotlib_8***

# In[47]:


# Personalizando colores
import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler

mpl.rc('lines',lw=2.)
mpl.rc('axes',facecolor='k',edgecolor='r')
mpl.rc('axes',prop_cycle=(cycler('color',['b','g','y'])))
mpl.rc('xtick',color='w')
mpl.rc('ytick',color='w')
mpl.rc('text',color='w')
mpl.rc('font',size=10)
mpl.rc('grid',color='g')
mpl.rc('grid',lw=1.5)
mpl.rc('figure',facecolor='b',edgecolor='w')


# In[48]:


X=np.linspace(0,7,1024)
plt.plot(X,np.sin(X))
plt.plot(X,np.cos(X))
plt.plot(X,np.cos(X)+np.sin(X))
plt.title('Osciloscopio')
plt.grid()
plt.show()


# In[49]:


plt.plot(X,X**2)
plt.plot(X,X**3)
plt.grid()
plt.title('Osciloscopio')
plt.show()


# ***Matplotlib_9***

# In[50]:


# Anotaciones

X=np.linspace(-4,4,512)
Y=.25*(X+4.)*(X+1.)*(X-2.)
plt.title('Curva polinomica_1')
plt.plot(X,Y,c='r',marker='*',ls='dotted')
plt.show()


# In[51]:


# Usando titulo con Latex
plt.title('$f(x)=\\frac{1}{4}(x+4)(x+1)(x-2)$')
plt.plot(X,Y,c='b')
plt.show()


# In[52]:


# Etiquetar los ejes

plt.title('Curva de una aleta de un aeroplano KV873')
plt.xlabel('velolicidad del aire')
plt.ylabel('resistencia total')
plt.plot(X,Y,c='b')
plt.show()


# In[53]:


# Texto en el medio de la grafica

plt.text(-0.5,-0.25,'El minimo')
plt.plot(X,Y,c='r')
plt.show()


# In[54]:


# Resaltando el texto

box={
    'facecolor':'r',
    'edgecolor':'g',
    'alpha':0.9,
    'boxstyle':'round'
}
plt.text(-0.1,-0.20,'minimo',bbox=box)# box define varios parmetros
plt.plot(X,Y,c='b')
plt.show()


# In[55]:


# Añadiendo flechas

plt.annotate('minimo',ha='center',va='bottom',xytext=(-1.5,3.),xy=(0.75,-2.7),arrowprops={'facecolor':'r',
            'shrink':0.005,'ArrowStyle':'fancy'})
plt.plot(X,Y,c='r')
plt.show()


# In[56]:


# Anadiendo leyendas

X=np.linspace(0,6,1024)
Y1=np.sin(X)
Y2=np.cos(X)
plt.title('Ejemplo de leyenda')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X,Y1,c='b',lw=3.,label='sin(X)')
plt.plot(X,Y2,c='r',lw=3.,ls='--',label='cos(X)')
plt.legend(shadow=True,loc='upper right')
plt.show()


# In[57]:


# Añadiendo cuadricula
X=np.linspace(-4,4,1024)
Y=.25*(X+4)*(X+1)*(X-2)
plt.plot(X,Y,c='r')
plt.grid(True,lw=1,ls='--',c='.75')
plt.show()


# In[58]:


# Escalas 
N=16
for i in range(N):
    plt.gca().add_line(plt.Line2D((0,i),(N-i,0),color='.75'))
plt.grid(True)
plt.axis('scaled')
plt.show()


# In[59]:


# Escalamiento extrecho

plt.gca().add_line(plt.Line2D((0,i),(5,5),color='0.2'))
plt.axis('tight')
plt.show()


# In[60]:


# Formas y fuguras
import matplotlib.patches as patches
import matplotlib.pyplot as plt 

forma=patches.Circle((0,0),radius=1.,color='.75')
plt.gca().add_patch(forma)

forma=patches.Rectangle((2.5,-.5),2.,1.,color='.75')
plt.gca().add_patch(forma)

forma=patches.Ellipse((0,-2),2.,1.,angle=45.,color='.75')
plt.gca().add_patch(forma)

forma=patches.FancyBboxPatch((2.5,-2.5),2.,1.,boxstyle='sawtooth',color='.75')
plt.gca().add_patch(forma)

plt.grid(True)
plt.axis('scaled')
plt.show()


# In[61]:


# Poligonos
theta=np.linspace(0,-2*np.pi,8)
puntos=np.vstack((np.cos(theta),np.sin(theta))).T
plt.gca().add_patch(patches.Polygon(puntos,color='.75'))
plt.gca().add_patch(plt.Polygon(puntos,closed=None,fill=None,lw=3.,ls='dashed',edgecolor='k'))   
plt.grid(True)
plt.axis('scaled')
plt.show()


# In[62]:


# Atributos en figuras
import matplotlib.ticker as ticker
X=np.linspace(-15,15,1024)
Y=np.sinc(X)
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
plt.plot(X,Y,c='r')
plt.show()


# In[63]:


lista_nombres=('Omar','Serguey','Max','Zhou','Alex')
lista_valores=np.random.randint(0,99,size=len(lista_nombres))
pos_lista=np.arange(len(lista_nombres))
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.FixedLocator((pos_lista)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter((lista_nombres)))
plt.bar(pos_lista,lista_valores,color='r',align='center',edgecolor='b')
plt.show()


# In[64]:


plt.bar(pos_lista,lista_valores,color='r',align='center',edgecolor='b')
plt.xticks(pos_lista,lista_nombres) # Se obtiene la misma funcion que con ax.xaxis 
plt.show()


# In[65]:


import datetime

fecha_inicio=datetime.datetime(2020,1,1)
def make_label(valor,pos):
    time=fecha_inicio+datetime.timedelta(days=365*valor)
    return time.strftime('%b %y')
ax=plt.axes()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_label))
X=np.linspace(0,1,256)
plt.plot(X,np.exp(-10*X),c='r',ls='--')
plt.plot(X,np.exp(-5*X),c='b',ls='--')
labels=ax.get_xticklabels()
plt.setp(labels,rotation=30.)
plt.show()


# ***Matplotlib_10***

# In[66]:


T= np.linspace(-np.pi,np.pi,1024)
tama_cuadric=(4,2)

plt.subplot2grid(tama_cuadric,(0,0),rowspan=3,colspan=1)
plt.plot(np.sin(2*T),np.cos(0.5*T),c='k')
plt.subplot2grid(tama_cuadric,(0,1),rowspan=3,colspan=1)
plt.plot(np.cos(5*T),np.sin(T),c='k')
plt.subplot2grid(tama_cuadric,(3,0),rowspan=1,colspan=3)
plt.plot(np.cos(5*T),np.sin(7*T),c='k')
plt.tight_layout()
plt.show()


# In[67]:


fig,ax0=plt.subplots(ncols=2,nrows=2)
ax0[0,0].plot(np.sin(2*T),np.cos(0.5*T),c='k')
ax0[1,0].plot(np.cos(3*T),np.sin(T),c='k')
plt.show()


# In[68]:


# Como usar escalas

plt.plot(2.*np.cos(T),np.sin(T),c='k',lw=3.)
plt.axes().set_aspect(2) # Hace que los ejes esten en la misma escala
plt.grid()
plt.show()


# In[69]:


# Rango de ejes

X=np.linspace(-6,6,1024)
plt.ylim(-0.1,1.5)
plt.plot(X,np.sinc(X),c='k')
plt.show()


# In[70]:


# Escalar valores 

Y1,Y2=np.sinc(X),np.cos(X)
plt.figure(figsize=(10.24,2.56))
plt.plot(X,Y1,c='k',lw=3.)
plt.plot(X,Y2,c='.75',lw=3.)
plt.show()


# In[71]:


# Resaltando figuras en tufigura

Y=np.sinc(X)
plt.plot(X,Y,c='k')
plt.grid()
X_en_detalle=np.linspace(-2,2,1024)
Y_en_detalle=np.sinc(X_en_detalle)
sub_axes=plt.axes([.6,.6,.25,.25])
sub_axes.plot(X_en_detalle,Y_en_detalle,c='b')
plt.grid(True,c='r')
plt.show()


# In[72]:


# Usando escalas logaritmicas

X=np.linspace(1,10,1024)
plt.yscale('log') # Sirve para escalar los datos todos a una misma escala
plt.plot(X,X,c='k',lw=2.,label=r'$f(x)=x$')
plt.plot(X,10**X,c='.75',ls='-.',lw=2.,label=r'$f(x)=e^x$')
plt.plot(X,np.log(X),c='.95',lw=2.,label=r'$f(X)=\log(x)$')
plt.legend()
plt.show()


# In[73]:


# Figuras en 3D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

fig=plt.figure()
ax=plt.axes(projection='3d')


# In[74]:


ax=plt.axes(projection='3d')
z_linea=np.linspace(0,15,1000)
x_linea=np.sin(z_linea)
y_linea=np.cos(z_linea)
ax.plot3D(x_linea,y_linea,z_linea,'gray')
# Datos para graficar usando scatter points
zdatos=15*np.random.random(100)
xdatos=np.sin(zdatos)+0.1*np.random.randn(100)
ydatos=np.cos(zdatos)+0.1*np.random.randn(100)
ax.scatter3D(xdatos,ydatos,zdatos,c=zdatos,cmap='viridis')
plt.show()


# In[75]:


# Contornos en 3D

def Graf_funtion(x,y):
    return np.sin(np.sqrt(x**2+y**2))
x=np.linspace(-6,6,30)
y=np.linspace(-6,6,30)
X,Y=np.meshgrid(x,y) # Funciona como plano para dibujar
Z=Graf_funtion(X,Y)

fig=plt.figure()
ax=plt.axes(projection='3d')
ax.contour3D(X,Y,Z,50,cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# In[76]:


ax.view_init(45,10)
fig


# In[77]:


# Wireframes
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_wireframe(X,Y,Z,color='black')
ax.set_title('wireframe')


# In[78]:


# Superficies
ax=plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='blue')
ax.set_title('superficies');


# In[79]:


r=np.linspace(0,6,20)
theta=np.linspace(-0.9*np.pi,0.8*np.pi,40)
r,theta=np.meshgrid(r,theta)
X=r*np.sin(theta)
Y=r*np.cos(theta)
Z=Graf_funtion(X,Y)
ax=plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none');


# In[80]:


# Usando scatter

theta=2*np.pi*np.random.random(1000)
r=6*np.random.random(1000)
x=r*np.sin(theta)
y=r*np.cos(theta)
z=Graf_funtion(x,y)
ax=plt.axes(projection='3d')
ax.scatter(x,y,z,cmap='viridis',lw=0.5);


# In[81]:


# Dando forma triangular a las figuras

ax=plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap='binary',edgecolor='red');

