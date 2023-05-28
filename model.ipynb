{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6df9f1fd",
   "metadata": {},
   "source": [
    "# Simple Handwritten Digit Classifier"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cffe4e1",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Load and view the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "549357da",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import mnist\n",
    "from tensorflow.keras import layers \n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "68d6305c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# The data has already been splitted into train and test set\n",
    "(train_data, train_labels), (test_data, test_labels) = mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5be75183",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training sample:\n",
      "[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136\n",
      "  175  26 166 255 247 127   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253\n",
      "  225 172 253 242 195  64   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251\n",
      "   93  82  82  56  39   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119\n",
      "   25   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253\n",
      "  150  27   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252\n",
      "  253 187   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249\n",
      "  253 249  64   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253\n",
      "  253 207   2   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253\n",
      "  250 182   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201\n",
      "   78   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]\n",
      " [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0\n",
      "    0   0   0   0   0   0   0   0   0   0]]\n",
      "\n",
      "Training labels:\n",
      "5\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Show the first training example\n",
    "print(f\"Training sample:\\n{train_data[0]}\\n\")\n",
    "print(f\"Training labels:\\n{train_labels[0]}\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4c27b9f6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((28, 28), ())"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the shape of a single example\n",
    "train_data[0].shape, train_labels[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2e7b4d1e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot a single sample\n",
    "plt.imshow(train_data[10]);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3cde1b6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check out sample label\n",
    "train_labels[10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "814a98a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, '4')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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XdqwYYM6Ozs1ffp0dXd3W0/NzJX//oP93JCk0tJSvfzyy9q9e7eef/55HThwQPfee6+i0aj11FLCOaeamhrNmDFDEydOlDQ4z4e+joOUOedDv/sqh2v5+vcLOed6rRvISktLY7+eNGmSpk2bpu9973vauHGjampqDGdmb7CfG5K0YMGC2K8nTpyou+66S4WFhdqxY4fKy8sNZ5YaS5Ys0UcffaT33nuv13OD6Xz4puOQKedDRlwJjRkzRkOGDOn1fzJdXV29/o9nMBk1apQmTZqko0ePWk/FzJV3B3Ju9BYKhVRYWDggz4+lS5fqzTff1J49e+K+f2ywnQ/fdBz60l/Ph4yI0PDhwzVlyhQ1NjbGrW9sbNT06dONZmUvGo3qk08+USgUsp6KmaKiIgWDwbhz4/z582pubh7U54YkdXd3q729fUCdH845LVmyRK+//rp2796toqKiuOcHy/lwvePQl357Phi+KcKTV155xQ0bNsz95S9/cR9//LGrrq52o0aNcseOHbOeWto89dRTrqmpyX322Wdu//797oEHHnBZWVkD/hj09PS41tZW19ra6iS5NWvWuNbWVvef//zHOefcqlWrXCAQcK+//ro7dOiQe/TRR10oFHKRSMR45sl1rePQ09PjnnrqKdfS0uLa2trcnj173LRp09wtt9wyoI7Dk08+6QKBgGtqanIdHR2x5auvvoptMxjOh+sdh0w6HzImQs4598ILL7jCwkI3fPhwd+edd8a9HXEwWLBggQuFQm7YsGEuPz/flZeXu8OHD1tPK+X27NnjJPVaKioqnHOX35a7YsUKFwwGnd/vd/fcc487dOiQ7aRT4FrH4auvvnIlJSVu7NixbtiwYe7WW291FRUV7vjx49bTTqq+/vySXENDQ2ybwXA+XO84ZNL5wPcJAQDMZMRrQgCAgYkIAQDMECEAgBkiBAAwQ4QAAGaIEADADBECAJghQgAAM0QIAGCGCAEAzBAhAICZ/wPE7aUlxqaRxgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot an example image and its label\n",
    "index_of_choice = 142\n",
    "plt.imshow(train_data[index_of_choice], cmap=plt.cm.binary)\n",
    "plt.title(train_labels[index_of_choice])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0a072a31",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjAAAAJDCAYAAAAVRy4AAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAcMElEQVR4nO3de7DWVbnA8fVuLsol3CJIoRBmjowZKZN4CZVJSsQRFTQrb6ODIGimgSimAkpeKkCdCmVUUsskhBE1DRVNcFQGUUYLzFIDRx1vyG2TYvKeP86cZjziu17ZL/u3H/bn8yfrmbWfGQu/5wetUyqXy+UEABBIXdELAAB8XgIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCETAAQDgCpoVbtmxZOvroo1PPnj1Tu3btUufOndPBBx+cfve73xW9GkBav359GjduXPrud7+bunbtmkqlUpo4cWLRa9EMCJgWbs2aNalHjx7pqquuSg888EC6/fbbU69evdKpp56aJk+eXPR6QAv33nvvpRkzZqQPP/wwHXfccUWvQzNS8v8LiS056KCD0htvvJFWrVpV9CpAC/Z//4oqlUrp3XffTV27dk0TJkzwFQZfYNiyLl26pNatWxe9BtDClUqlVCqVil6DZsi/oUgppbR58+a0efPm9P7776fZs2en+fPnp1/96ldFrwUAWyRgSCmlNHr06HTTTTellFJq27ZtuuGGG9LIkSML3goAtkzAkFJK6ZJLLknDhw9Pb7/9drrvvvvSueeemxoaGtLYsWOLXg0APkXAkFJKqWfPnqlnz54ppZQGDx6cUkpp/Pjx6fTTT09du3YtcjUA+BR/iZct6tevX/rPf/6TXnnllaJXAYBPETBs0WOPPZbq6urSV77ylaJXAYBP8UdILdyIESNSp06dUr9+/VK3bt3Su+++m2bPnp1mzZqVLrzwQn98BBTuwQcfTA0NDWn9+vUppZSWL1+e7r777pTS//6Rd/v27Ytcj4J4yK6FmzlzZpo5c2ZasWJFWrNmTerYsWP6xje+kYYPH55OOeWUotcDSL169UorV67c4tmrr76aevXq1bQL0SwIGAAgHH8HBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAhHwAAA4bQueoHm7OOPP87ObNy4MTtz6623Vjx/5513snc888wz2Zk+ffpkZ1q3rvyPvK6u6Zr2xBNPrHi+7777Zu9o1apVrdYBtgPTpk3LzowZMyY7s3nz5lqswzbkCwwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcDxkV8GTTz6ZnRkwYECjf0779u2zM7kH6FJK6fnnn8/ObNiwoeJ5Q0ND9o5aufrqqyueDxs2LHvH1KlTszO777571TsBsf3973/PzpRKpSbYhG3NFxgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAjHOzAVtGvXLjuz1157ZWfOOOOMiuff+ta3snd07ty5Jrvk3op54YUXsne8//772ZkbbrghO7Nq1aqK53PmzMne0adPn+zMpZdemp0Btg+PP/54dubQQw9tgk3Y1nyBAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABBOqVwul4tegu3PmjVrsjM//OEPK57Pnz8/e0ePHj2yM08//XR25otf/GJ2BijWihUrsjNf+9rXsjMjR47MzkyfPr2qnSiOLzAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABCOgAEAwmld9AJsn+rr67Mze+65Z6N/zh577JGd6dSpU6N/DlC8OXPmZGe8zdpy+AIDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhCBgAIBwP2fG5NTQ0ZGd+8pOfZGfmzp3b6F3GjBmTnWnfvn2jfw5QvHnz5mVnSqVSE2xCc+ALDAAQjoABAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhOMdGD5h0aJF2ZlRo0ZlZ1asWNHoXb797W9nZ/bdd99G/xxg+1Eul4tegSbiCwwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcDxktx254447sjMLFiyoeH7vvfdm71i7dm12ZpdddsnOnHPOORXPx40bl72jXbt22Rmg5SiVStmZ448/vgk2YVvzBQYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOB6yawZuv/327MzkyZOzM6+++mp25uOPP654Xs0jUPX19dmZuXPnZmf69++fnQH4PMrlcnamS5cuTbAJ25ovMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDC8ZBdI02cODE7M23atIrnGzduzN6xefPmaldqlKFDh2ZnbrvttuxM+/bta7EOwOdSzWOcbB98gQEAwhEwAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHC8A9MENm3aVPG8mjdeyuVyTXbJ3TNnzpzsHR9++GF2ZuDAgdmZ4cOHVzz3lgy0LCtXrqx4vmrVquwdPXv2rMkMzZ8vMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCKZVr9UIan+mZZ56peL5ixYrsHWvXrs3OTJkyJTuTe4Ruw4YN2TsaGhqyM9XYbbfdKp6PGTMme8fo0aOzM23atKl6J6A4S5curXjer1+/7B19+/bNzixZsqTqnWi+fIEBAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6H7PiExYsXZ2eqeTDvoYceys6sX7++qp0queiii7IzEydOzM60bdu20bsAjXP22WdXPJ8xY0b2jhtvvDE7M2LEiKp3ovnyBQYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOK2LXoDm5cADD8zO/PGPf8zOPP3009mZqVOnVjyfP39+9o5rr702O1ONSZMmVTxv06ZNTX4OsPVKpVLRK9CM+AIDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhlMrlcrnoJT6v1157LTvTo0ePJtiEbenOO+/MzowaNSo7s2HDhuzMuHHjKp5Pnjw5e0erVq2yM8Bny73zUs07MDfeeGN2ZsSIEVXvRPPlCwwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcJrdQ3abNm3KzgwaNCg706lTp4rnY8eOzd7Rv3//7AzFmjZtWnammn/WOevWrcvOdOjQodE/B1qyurrK/zd1NQ/ZTZ8+PTvjIbvtgy8wAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMJpXfQC/1/btm2zMyeeeGJ25txzz614vmDBguwdRxxxRHbm5JNPzs585zvfqXheX1+fvYMte/jhh2tyz1577VXxvFWrVjX5OcBnq8W7qs3sbVa2IV9gAIBwBAwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAITT7B6yq8ZZZ52Vndm0aVPF88cffzx7x7x587Iz9913X3amR48eFc8POOCA7B2DBw/OzlSjb9++Fc9zD7qllNKsWbOyM2+99VZ2ZubMmdmZnFWrVjX6jpRSGjduXMXzHXfcsSY/B/hspVKpUefVzrB98AUGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCKZXL5XLRSxTho48+ys6sXr06O/Pyyy9nZ2bPnl3xfPny5dk7HnnkkexMNTp27FjxvH379tk73n777Zrs0lR++ctfZmfOO++8iuetWrWq1TrAZ6jFOzDTp0/PzowcObLqnWi+fIEBAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEE6LfciuOanmUb0nn3wyO1PNY3d/+ctfGv1zqjF06NDsTO/evSue77HHHtk7TjrppOxMu3btsjN1dVoeinbZZZdVPL/qqquyd+R+j0sppUMPPbTalWjG/K4NAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhCBgAIBwP2QEA4fgCAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGFJKKT3xxBNp8ODBaeedd07t2rVLe+21V7ryyiuLXgto4Z577rl03HHHpe7du6f27dun3r17pyuuuCJt3Lix6NUoWOuiF6B4d955Zzr11FPT9773vXT77benjh07ppdffjm98cYbRa8GtGDLly9PhxxySNp7773Tddddl7p06ZIWLlyYrrjiirR06dI0b968olekQKVyuVwuegmK8/rrr6e99947nXbaaek3v/lN0esA/Nell16afvazn6V//vOfac899/zvr48cOTLNmDEjrV69Ou28884FbkiR/BFSC3fzzTenhoaGdNFFFxW9CsAntGnTJqWU0k477fSJX6+vr091dXWpbdu2RaxFMyFgWriFCxemzp07pxdffDHtt99+qXXr1mnXXXdNZ599dlq3bl3R6wEt2Omnn57q6+vTqFGj0iuvvJLWr1+f7r///nTTTTelc845J3Xo0KHoFSmQP0Jq4Xr37p1WrlyZ2rRpk8aPH58OPvjgtGTJkjRhwoTUt2/ftGjRolQqlYpeE2ihXnzxxXT88cenF1988b+/dt5556XrrrvO700tnL/E28Jt3rw5ffDBB2nChAnp4osvTimlNGDAgNS2bdt0/vnnpwULFqSBAwcWvCXQEv3rX/9KxxxzTOrWrVu6++67U9euXdPixYvT5MmT04YNG9Itt9xS9IoUyB8htXC77LJLSimlI4888hO/ftRRR6WUUnr22WebfCeAlFK6+OKL07p169L8+fPTsGHD0mGHHZYuvPDCdN1116Vbb701Pf7440WvSIEETAvXp0+fLf76//3JYl2d/4gAxVi2bFnaZ599PvV3XQ444ICUUkp//etfi1iLZsK/nVq4YcOGpZRSevDBBz/x6w888EBKKaWDDjqoyXcCSCml7t27p7/97W9pw4YNn/j1p556KqWU0u67717EWjQT/hIvaciQIemhhx5Kl156aTrooIPSM888kyZNmpQGDhyY7rvvvqLXA1qoe++9Nx133HHpwAMPTBdccEHq0qVLevrpp9PVV1+devbsmZ577jn/U+oWTMCQ/v3vf6dJkyalO++8M7355pupe/fu6eSTT04TJkxIO+ywQ9HrAS3YY489lq655pr0/PPPp7Vr16YePXqkY445Jo0fP/6/f4ePlknAAADh+DswAEA4AgYACEfAAADhCBgAIBwBAwCEI2AAgHAEDAAQjoABAMIRMABAOAIGAAhHwAAA4QgYACAcAQMAhCNgAIBwBAwAEI6AAQDCETAAQDgCBgAIR8AAAOEIGAAgHAEDAIQjYACAcAQMABCOgAEAwhEwAEA4AgYA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",
      "text/plain": [
       "<Figure size 700x700 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot multiple random images of fashion mnist\n",
    "import random\n",
    "plt.figure(figsize=(7, 7))\n",
    "for i in range(4):\n",
    "  ax = plt.subplot(2, 2, i+1)\n",
    "  rand_index = random.choice(range(len(train_data)))\n",
    "  plt.imshow(train_data[rand_index], cmap=plt.cm.binary)\n",
    "  plt.title(train_labels[rand_index])\n",
    "  plt.axis(False);"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9cae4cae",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Building a multiclass classification model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1cec6dba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0, 255)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the min and max of train data\n",
    "train_data.min(), train_data.max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f7f5a2de",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalize train/test data\n",
    "train_data_norm = train_data / 255.0\n",
    "test_data_norm = test_data / 255.0"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c9f95b4c",
   "metadata": {},
   "source": [
    "### Base model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5b3c6b17",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Metal device set to: Apple M1\n",
      "Epoch 1/5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-05-28 17:58:13.870871: W tensorflow/tsl/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1875/1875 [==============================] - 11s 6ms/step - loss: 1.0019 - accuracy: 0.6753 - val_loss: 0.6765 - val_accuracy: 0.7884\n",
      "Epoch 2/5\n",
      "1875/1875 [==============================] - 10s 5ms/step - loss: 0.6335 - accuracy: 0.8099 - val_loss: 0.6009 - val_accuracy: 0.8215\n",
      "Epoch 3/5\n",
      "1875/1875 [==============================] - 10s 6ms/step - loss: 0.5832 - accuracy: 0.8275 - val_loss: 0.5668 - val_accuracy: 0.8345\n",
      "Epoch 4/5\n",
      "1875/1875 [==============================] - 10s 5ms/step - loss: 0.5527 - accuracy: 0.8410 - val_loss: 0.5480 - val_accuracy: 0.8441\n",
      "Epoch 5/5\n",
      "1875/1875 [==============================] - 11s 6ms/step - loss: 0.5331 - accuracy: 0.8482 - val_loss: 0.5319 - val_accuracy: 0.8522\n"
     ]
    }
   ],
   "source": [
    "# Set random seed\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Create a model\n",
    "model_1 = tf.keras.Sequential([\n",
    "    layers.Flatten(input_shape=(28,28)),\n",
    "    layers.Dense(4, activation=\"relu\"),\n",
    "    layers.Dense(10, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "                 optimizer=tf.keras.optimizers.legacy.Adam(),\n",
    "                 metrics=[\"accuracy\"])\n",
    "\n",
    "# Fit the model\n",
    "history_1 = model_1.fit(train_data_norm,\n",
    "                       train_labels,\n",
    "                       epochs=5,\n",
    "                       validation_data=(test_data_norm, test_labels))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ab7ad48",
   "metadata": {},
   "source": [
    "### Second model - try CNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8efe2006",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "1875/1875 [==============================] - 19s 10ms/step - loss: 0.2076 - accuracy: 0.9401 - val_loss: 0.0964 - val_accuracy: 0.9702\n",
      "Epoch 2/5\n",
      "1875/1875 [==============================] - 17s 9ms/step - loss: 0.0790 - accuracy: 0.9775 - val_loss: 0.0656 - val_accuracy: 0.9782\n",
      "Epoch 3/5\n",
      "1875/1875 [==============================] - 18s 10ms/step - loss: 0.0603 - accuracy: 0.9824 - val_loss: 0.0574 - val_accuracy: 0.9809\n",
      "Epoch 4/5\n",
      "1875/1875 [==============================] - 15s 8ms/step - loss: 0.0499 - accuracy: 0.9848 - val_loss: 0.0564 - val_accuracy: 0.9828\n",
      "Epoch 5/5\n",
      "1875/1875 [==============================] - 16s 8ms/step - loss: 0.0421 - accuracy: 0.9870 - val_loss: 0.0539 - val_accuracy: 0.9827\n"
     ]
    }
   ],
   "source": [
    "# Set random seed\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Create a model\n",
    "model_2 = tf.keras.Sequential([\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(10, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model_2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "                 optimizer=tf.keras.optimizers.legacy.Adam(),\n",
    "                 metrics=[\"accuracy\"])\n",
    "\n",
    "# Fit the model\n",
    "history_2 = model_2.fit(train_data_norm,\n",
    "                        train_labels,\n",
    "                        epochs=5,\n",
    "                        validation_data=(test_data_norm, test_labels))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "865ca7e3-1145-43e2-8624-8b01f19234fa",
   "metadata": {},
   "source": [
    "### Let's combine these models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "98afc809",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.1438 - accuracy: 0.9558 - val_loss: 0.0528 - val_accuracy: 0.9825\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.0495 - accuracy: 0.9844 - val_loss: 0.0351 - val_accuracy: 0.9880\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0337 - accuracy: 0.9894 - val_loss: 0.0276 - val_accuracy: 0.9923\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0250 - accuracy: 0.9922 - val_loss: 0.0322 - val_accuracy: 0.9906\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0206 - accuracy: 0.9934 - val_loss: 0.0352 - val_accuracy: 0.9900\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0157 - accuracy: 0.9948 - val_loss: 0.0362 - val_accuracy: 0.9910\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.0151 - accuracy: 0.9951 - val_loss: 0.0343 - val_accuracy: 0.9901\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0118 - accuracy: 0.9959 - val_loss: 0.0420 - val_accuracy: 0.9907\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0090 - accuracy: 0.9972 - val_loss: 0.0370 - val_accuracy: 0.9901\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0096 - accuracy: 0.9970 - val_loss: 0.0426 - val_accuracy: 0.9896\n"
     ]
    }
   ],
   "source": [
    "# Set random seed\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Create a model\n",
    "model_3 = tf.keras.Sequential([\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(10, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model_3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "                 optimizer=tf.keras.optimizers.legacy.Adam(),\n",
    "                 metrics=[\"accuracy\"])\n",
    "\n",
    "# Fit the model\n",
    "history_3 = model_3.fit(train_data_norm,\n",
    "                        train_labels,\n",
    "                        epochs=10,\n",
    "                        validation_data=(test_data_norm, test_labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e2d0643d-b853-477f-9430-97719394d481",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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CBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQlVqzCyfPlyREdHw2g0IjY2Fnv37q22/4YNG9CzZ0/4+fkhIiICf/rTn5CXl1ergomIiKhx8TqMbN68GbNnz8bTTz+NI0eOYODAgRg2bBgyMzM99t+3bx8mTpyIKVOm4Pjx49iyZQsOHTqEqVOnXnHxRERE1PB5HUaWLFmCKVOmYOrUqYiJicHSpUsRFRWFFStWeOz/zTffoH379pg5cyaio6Nx880347HHHsPhw4evuHgiIiJq+LwKI2azGampqUhISHBrT0hIwIEDBzxuEx8fj/Pnz2P79u1QFAUXL17Ev/71L9x11121r5qIiIgaDa/CSG5uLmw2G8LDw93aw8PDkZ2d7XGb+Ph4bNiwAaNHj4aPjw9atWqFkJAQvPHGG1Uex2QyoaCgwO1GREREjVOtLmCVJMltWVGUSm1OJ06cwMyZM/Hcc88hNTUVX3zxBTIyMjBt2rQq95+cnIzg4GD1FhUVVZsyiYiIqAGQFEVRatrZbDbDz88PW7ZswX333ae2z5o1C2lpadi9e3elbSZMmICysjJs2bJFbdu3bx8GDhyICxcuICIiotI2JpMJJpNJXS4oKEBUVBTy8/MRFBRU4ztHRERE4hQUFCA4OPiyz99ejYz4+PggNjYWKSkpbu0pKSmIj4/3uE1JSQk0GvfDaLVaAPYRFU8MBgOCgoLcbkRERNQ4eX2aJjExEe+88w7WrFmD9PR0zJkzB5mZmeppl6SkJEycOFHtP2LECHz88cdYsWIFfvnlF+zfvx8zZ87EjTfeiMjIyLq7J0RERNQg6bzdYPTo0cjLy8PChQuRlZWFbt26Yfv27WjXrh0AICsry+0zRyZPnozCwkIsW7YMTz75JEJCQjB48GAsWrSo7u4FERERNVheXTMiSk3POREREVH9cVWuGSEiIiKqawwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUAwjREREJBTDCBEREQnFMEJERERCMYwQERGRUDrRBRARUf1ks9lgsVhEl0H1mF6vh1arveL9MIwQEZEbRVGQnZ2NP/74Q3Qp1ACEhISgVatWkCSp1vtgGCEiIjfOIBIWFgY/P78repKhxktRFJSUlCAnJwcAEBERUet9MYwQEZHKZrOpQSQ0NFR0OVTP+fr6AgBycnIQFhZW61M2vICViIhUzmtE/Pz8BFdCDYXzd+VKri9iGCEiokp4aoZqqi5+VxhGiIiISCiGESIiahRuvfVWzJ49W3QZVAu1CiPLly9HdHQ0jEYjYmNjsXfv3mr7m0wmPP3002jXrh0MBgOuu+46rFmzplYFExERUePi9btpNm/ejNmzZ2P58uUYMGAAVq5ciWHDhuHEiRNo27atx20eeughXLx4EatXr0bHjh2Rk5MDq9V6xcUTERFRw+f1yMiSJUswZcoUTJ06FTExMVi6dCmioqKwYsUKj/2/+OIL7N69G9u3b8ftt9+O9u3b48Ybb0R8fPwVF09EROTJpUuXMHHiRDRr1gx+fn4YNmwYfvrpJ3X92bNnMWLECDRr1gz+/v7o2rUrtm/frm47btw4tGzZEr6+vujUqRPWrl0r6q40CV6NjJjNZqSmpmL+/Plu7QkJCThw4IDHbbZu3Yq4uDi88sorWL9+Pfz9/XHPPffghRdeUN+fTERE9ZeiKCi12IQc21evrdW7NSZPnoyffvoJW7duRVBQEObNm4fhw4fjxIkT0Ov1mD59OsxmM/bs2QN/f3+cOHECAQEBAIBnn30WJ06cwOeff44WLVrg559/RmlpaV3fNXLhVRjJ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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.DataFrame(history_3.history).plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "642bcca9-c5b8-4c5e-b951-88f3bcbb47f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.4477 - accuracy: 0.8505 - val_loss: 0.1056 - val_accuracy: 0.9651\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.1708 - accuracy: 0.9470 - val_loss: 0.0698 - val_accuracy: 0.9769\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 27s 14ms/step - loss: 0.1310 - accuracy: 0.9578 - val_loss: 0.0885 - val_accuracy: 0.9698\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.1106 - accuracy: 0.9654 - val_loss: 0.0604 - val_accuracy: 0.9803\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.0996 - accuracy: 0.9690 - val_loss: 0.0506 - val_accuracy: 0.9844\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.0912 - accuracy: 0.9716 - val_loss: 0.0625 - val_accuracy: 0.9793\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.0859 - accuracy: 0.9729 - val_loss: 0.0496 - val_accuracy: 0.9833\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 25s 13ms/step - loss: 0.0807 - accuracy: 0.9747 - val_loss: 0.0449 - val_accuracy: 0.9842\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 26s 14ms/step - loss: 0.0789 - accuracy: 0.9751 - val_loss: 0.0419 - val_accuracy: 0.9860\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 24s 13ms/step - loss: 0.0716 - accuracy: 0.9773 - val_loss: 0.0468 - val_accuracy: 0.9846\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "# Create an instance of ImageDataGenerator for data augmentation\n",
    "datagen = ImageDataGenerator(\n",
    "    rotation_range=10,  # Randomly rotate images within the range of 10 degrees\n",
    "    width_shift_range=0.1,  # Randomly shift images horizontally by a fraction of the width\n",
    "    height_shift_range=0.1,  # Randomly shift images vertically by a fraction of the height\n",
    "    zoom_range=0.1,  # Randomly zoom images by a factor of 0.1\n",
    "    horizontal_flip=True  # Randomly flip images horizontally\n",
    ")\n",
    "\n",
    "# Fit the ImageDataGenerator to the training data\n",
    "datagen.fit(train_data_norm.reshape(-1, 28, 28, 1))\n",
    "\n",
    "# Set random seed\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Create a model\n",
    "model_4 = tf.keras.Sequential([\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\"),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(10, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model_4.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "                 optimizer=tf.keras.optimizers.legacy.Adam(),\n",
    "                 metrics=[\"accuracy\"])\n",
    "\n",
    "# Fit the model with augmented data\n",
    "history_4 = model_4.fit(datagen.flow(train_data_norm.reshape(-1, 28, 28, 1), train_labels, batch_size=32),\n",
    "                        epochs=10,\n",
    "                        validation_data=(test_data_norm.reshape(-1, 28, 28, 1), test_labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "667b8cad-9c8d-4bc5-8e60-a3133ebf64b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 22s 11ms/step - loss: 0.1500 - accuracy: 0.9535 - val_loss: 0.0644 - val_accuracy: 0.9788 - lr: 0.0010\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.0533 - accuracy: 0.9832 - val_loss: 0.0558 - val_accuracy: 0.9813 - lr: 0.0013\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.0436 - accuracy: 0.9865 - val_loss: 0.0381 - val_accuracy: 0.9870 - lr: 0.0016\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.0359 - accuracy: 0.9892 - val_loss: 0.0448 - val_accuracy: 0.9875 - lr: 0.0020\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.0385 - accuracy: 0.9880 - val_loss: 0.0324 - val_accuracy: 0.9901 - lr: 0.0025\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.0365 - accuracy: 0.9885 - val_loss: 0.0481 - val_accuracy: 0.9859 - lr: 0.0032\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.0458 - accuracy: 0.9871 - val_loss: 0.0446 - val_accuracy: 0.9880 - lr: 0.0040\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0516 - accuracy: 0.9863 - val_loss: 0.0736 - val_accuracy: 0.9812 - lr: 0.0050\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0609 - accuracy: 0.9843 - val_loss: 0.0892 - val_accuracy: 0.9761 - lr: 0.0063\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 24s 13ms/step - loss: 0.0771 - accuracy: 0.9814 - val_loss: 0.0777 - val_accuracy: 0.9821 - lr: 0.0079\n"
     ]
    }
   ],
   "source": [
    "# Data augmentation does not help, let's keep model 3\n",
    "# Set random seed\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Create a model\n",
    "model_5 = tf.keras.Sequential([\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(10, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model_5.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "                 optimizer=tf.keras.optimizers.legacy.Adam(),\n",
    "                 metrics=[\"accuracy\"])\n",
    "\n",
    "# Create a learning rate scheduler\n",
    "lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/10))\n",
    "\n",
    "# Fit the model\n",
    "history_5 = model_5.fit(train_data_norm,\n",
    "                        train_labels,\n",
    "                        epochs=10,\n",
    "                        validation_data=(test_data_norm, test_labels),\n",
    "                       callbacks=[lr_scheduler])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d15da4e5-b9b2-47d3-a0a5-aa581bb2020f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'finding the ideal learning rate')"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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adOmFi2QmkczBJZeAD71hIiIqEqTAtBff/2FgQMHAgC++eYbODk5ISEhAV988QU++eSTFi2Qmsfb3gIyqQQFpZVIySsVuxwiIiKt0KQAVFxcDEvLqp6Fw4cPY8KECZBKpejbty8SEhJatEBqHhOZFD4OFgA4EZqIiKhakwJQhw4d8P333yMpKQm//PILQkJCAAAZGRk6+/R0fVY9DBadynlAREREQBMD0Ouvv46XX34Znp6e6N27N4KDgwFU9Qb16NGjRQuk5uMjMYiIiGpq0m3wkyZNwoABA5CamqpZAwgAhg0bhkcffbTFiqOWwTvBiIiIampSAAIAZ2dnODs749atW5BIJGjXrh0XQdRS1T1ANzILUV6phomsSR1/REREeqNJn4RqtRqrVq2CUqmEh4cH3N3dYW1tjbfffhtqtbqla6RmamdtCku5DJVqATezCsUuh4iISHRN6gFavnw5duzYgffeew/9+/eHIAg4deoU3nzzTZSWluLdd99t6TqpGSQSCTo5WyIy4TZi0wrg58yJ6kREZNiaFIA+//xzbN++XfMUeAAIDAxEu3btMHfuXAYgLeR7JwDFpBVgnNjFEBERiaxJQ2A5OTnw8/Ortd3Pzw85OTnNLopaHidCExER/aNJASgwMBCffvppre2ffvopunXr1uyiqOX5OjEAERERVWvSENiaNWvw8MMP48iRIwgODoZEIsHp06eRlJSEsLCwlq6RWkD1vJ/k3BLkl1bASmEsckVERETiaVIP0ODBg3H16lU8+uijyM3NRU5ODiZMmIDLly/js88+a+kaqQUozYzhbKUAAFxlLxARERk4idCCjwiPiopCz549oVKpWmqXosjPz4dSqUReXp5ePdpj+s4zOH41E++MD8CUvh5il0NERNSiGvP5zRXxDAgnQhMREVVhADIgfi4MQERERAADkEHxdarqDoxJy0cLjnwSERHpnEbdBTZhwoR7vp6bm9ucWqiV+Tiaw0gqQX5pJdLyS+GiNBW7JCIiIlE0KgAplcr7vj5t2rRmFUStRy4zgre9Oa5lFCImrYABiIiIDFajAhBvcdd9vs6WuJZRiNi0Agz1dRS7HCIiIlFwDpCB4Z1gREREDEAGx9e5eiI0AxARERkuBiADU90DdCOjEBUqtcjVEBERiYMByMC0szaFuYkRylVqxGcViV0OERGRKBiADIxU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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "lrs = 1e-3 * (10**(tf.range(10)/10))\n",
    "plt.semilogx(lrs, history_5.history[\"loss\"])\n",
    "plt.xlabel(\"Learning rate\")\n",
    "plt.ylabel(\"Loss\")\n",
    "plt.title(\"finding the ideal learning rate\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c00a2b51-ada0-4fa7-8469-9f2f1fa0f187",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "1875/1875 [==============================] - 22s 11ms/step - loss: 0.1301 - accuracy: 0.9588 - val_loss: 0.0643 - val_accuracy: 0.9789\n",
      "Epoch 2/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.0493 - accuracy: 0.9848 - val_loss: 0.0571 - val_accuracy: 0.9828\n",
      "Epoch 3/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.0362 - accuracy: 0.9890 - val_loss: 0.0330 - val_accuracy: 0.9900\n",
      "Epoch 4/10\n",
      "1875/1875 [==============================] - 22s 12ms/step - loss: 0.0295 - accuracy: 0.9910 - val_loss: 0.0403 - val_accuracy: 0.9879\n",
      "Epoch 5/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.0229 - accuracy: 0.9928 - val_loss: 0.0374 - val_accuracy: 0.9901\n",
      "Epoch 6/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0191 - accuracy: 0.9940 - val_loss: 0.0584 - val_accuracy: 0.9861\n",
      "Epoch 7/10\n",
      "1875/1875 [==============================] - 24s 13ms/step - loss: 0.0188 - accuracy: 0.9941 - val_loss: 0.0440 - val_accuracy: 0.9882\n",
      "Epoch 8/10\n",
      "1875/1875 [==============================] - 23s 12ms/step - loss: 0.0162 - accuracy: 0.9953 - val_loss: 0.0425 - val_accuracy: 0.9901\n",
      "Epoch 9/10\n",
      "1875/1875 [==============================] - 20s 11ms/step - loss: 0.0150 - accuracy: 0.9957 - val_loss: 0.0384 - val_accuracy: 0.9910\n",
      "Epoch 10/10\n",
      "1875/1875 [==============================] - 21s 11ms/step - loss: 0.0138 - accuracy: 0.9959 - val_loss: 0.0431 - val_accuracy: 0.9903\n"
     ]
    }
   ],
   "source": [
    "# Data augmentation does not help, let's keep model 3\n",
    "# Set random seed\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "# Create a model\n",
    "model_6 = tf.keras.Sequential([\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Conv2D(filters=32, kernel_size=3, activation=\"relu\", input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(pool_size=2),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(128, activation=\"relu\"),\n",
    "    layers.Dense(10, activation=\"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model_6.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "                 optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.002),\n",
    "                 metrics=[\"accuracy\"])\n",
    "\n",
    "# Fit the model\n",
    "history_6 = model_6.fit(train_data_norm,\n",
    "                        train_labels,\n",
    "                        epochs=10,\n",
    "                        validation_data=(test_data_norm, test_labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2d4e16f9-9ebc-478a-9ad1-fc1f03e3397c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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5eVp165dSdeZOHGilixZoosuukgej0f9+vVT9+7d9Yc//KHV/cyfP1+5ubmxR35+fiplAgAAADjE7dPNCAzDSJi3LKtFW9S6des0e/Zs3XHHHVqzZo1ef/11bdmyRTNnzmx1+/PmzVNlZWXssW3btn0pEwAAAMAhypVK5969e8vpdLYYvSkrK2sxyhM1f/58nXDCCbrlllskSUceeaSysrJ00kkn6e6771b//v1brOP1euX1elMpDQAAAABiUhrR8Xg8KiwsVFFRUUJ7UVGRJk6cmHSduro6ORyJu3E6nZLCI0EAAAAAcKClfOra3Llz9ac//UlPPPGE1q9frxtvvFElJSWxU9HmzZunK6+8MtZ/+vTpeumll7R48WJt3rxZ7733nmbPnq3vfve7GjBgwIF7JQAAAAAQkdKpa5J00UUXqaKiQnfddZdKS0s1ZswYLV++XEOGDJEklZaWJnynzowZM1RdXa2FCxfqpptuUvfu3TVp0iTde++9B+5VAAAAAEAcw+oC549VVVUpNzdXlZWVysnJSXc5AAAAANKkvdlgn+66BgAAAAAHM4IOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh6AAAAACwHYIOAAAAANsh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QY0dO1ZZWVmaM2eO/H5/u/YrhU9fO/roo7V9+3Y98cQTmjx5soYMGbLX9boKbkYAAAAAtKKiokLr16/XL3/5S02ePFmjRo3Snj17YsuPPPJIFRcX65tvvkm6/pFHHql//vOfrW6/T58+Ki0tjc1v3LhRdXV1e61r5cqVOvvss3X55ZfrqKOO0rBhw7Rx48bY8hEjRigjI6PNfY8dO1bjx4/XY489pueee05XX331XvfblRB0AAAAgFb06NFDvXr10qOPPqovv/xSb775pubOnRtbfskll6hfv34655xz9N5772nz5s168cUX9f7770uSfv3rX+v555/Xr3/9a61fv16ffvqp7rvvvtj6kyZN0sKFC/Xxxx/ro48+0syZM+V2u/da1/Dhw1VUVKRVq1Zp/fr1+vGPf6xdu3bFlvt8Pt166636+c9/rqefflqbNm3SBx98oMcffzxhO9dee61+97vfKRQK6dxzz93ft+ugQtABAAAAWuFwOPSXv/xFa9as0ZgxY3TjjTfqv/7rv2LLPR6PVqxYob59+2ratGkaO3asfve738npdEqSTj31VL3wwgtatmyZjj76aE2aNCnh1tMLFixQfn6+Tj75ZF166aW6+eablZm592uIfvWrX2ncuHE644wzdOqpp8bCVvM+N910k+644w6NGjVKF110kcrKyhL6XHLJJXK5XLr00kvl8/n24506+BhW85MCD0JVVVXKzc1VZWWlcnJy0l0OAAAAUtDQ0KAtW7aooKDAdgfTXd22bds0dOhQrV69WuPGjUt3OTFt/cy0NxtwMwIAAADgEBMIBFRaWqrbbrtNxx133EEVcg4UTl0DAAAADjHvvfeehgwZojVr1ujhhx9OdzkdghEdAAAA4BBz6qmntrittd0wogMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAAB0kKFDh+rBBx9sV1/DMPTKK690aD2HEoIOAAAAANsh6AAAAACwHYIOAAAAOp1lWaoL1KXlYVlWu2p85JFHNHDgQJmmmdD+gx/8QFdddZU2bdqks88+W3l5eerWrZuOPfZY/eMf/zhg79Gnn36qSZMmKSMjQ7169dKPfvQj1dTUxJb/61//0ne/+11lZWWpe/fuOuGEE7R161ZJ0ieffKLvfe97ys7OVk5OjgoLC/XRRx8dsNq6Ale6CwAAAMChpz5YrwnPTUjLvj+89ENlujP32u+CCy7Q7Nmz9dZbb2ny5MmSpD179uiNN97Q3/72N9XU1GjatGm6++675fP59Oc//1nTp0/XF198ocGDB+9XjXV1dTrzzDN13HHHafXq1SorK9O1116rWbNm6amnnlIwGNQ555yj6667Ts8//7z8fr/+/e9/yzAMSdJll12mY445RosXL5bT6VRxcbHcbvd+1dTVEHQAAACAJHr27KkzzzxTzz33XCzovPDCC+rZs6cmT54sp9Opo446Ktb/7rvv1ssvv6xly5Zp1qxZ+7XvJUuWqL6+Xk8//bSysrIkSQsXLtT06dN17733yu12q7KyUmeddZYOO+wwSdKoUaNi65eUlOiWW27Rd77zHUnSiBEj9queroigAwAAgE6X4crQh5d+mLZ9t9dll12mH/3oR1q0aJG8Xq+WLFmiiy++WE6nU7W1tbrzzjv16quvaufOnQoGg6qvr1dJScl+17h+/XodddRRsZAjSSeccIJM09QXX3yhk08+WTNmzNAZZ5yh008/XaeddpouvPBC9e/fX5I0d+5cXXvttXrmmWd02mmn6YILLogFokMF1+gAAACg0xmGoUx3Zloe0dO72mP69OkyTVOvvfaatm3bppUrV+ryyy+XJN1yyy168cUXdc8992jlypUqLi7W2LFj5ff79/v9sSyr1Tqj7U8++aTef/99TZw4UUuXLtXIkSP1wQcfSJJ+85vf6LPPPtP3v/99vfnmmzriiCP08ssv73ddXQlBBwAAAGhFRkaGzjvvPC1ZskTPP/+8Ro4cqcLCQknSypUrNWPGDJ177rkaO3as+vXrp6+++uqA7PeII45QcXGxamtrY23vvfeeHA6HRo4cGWs75phjNG/ePK1atUpjxozRc889F1s2cuRI3XjjjVqxYoXOO+88Pfnkkwektq6CoAMAAAC04bLLLtNrr72mJ554IjaaI0nDhw/XSy+9pOLiYn3yySe69NJLW9yhbX/26fP5dNVVV+n//u//9NZbb+mGG27QFVdcoby8PG3ZskXz5s3T+++/r61bt2rFihXasGGDRo0apfr6es2aNUv/+te/tHXrVr333ntavXp1wjU8hwKu0QEAAADaMGnSJPXs2VNffPGFLr300lj7Aw88oKuvvloTJ05U7969deutt6qqquqA7DMzM1NvvPGGfvazn+nYY49VZmamzj//fN1///2x5Z9//rn+/Oc/q6KiQv3799esWbP04x//WMFgUBUVFbryyiv19ddfq3fv3jrvvPN05513HpDaugrDau+NxNOoqqpKubm5qqysVE5OTrrLAQAAQAoaGhq0ZcsWFRQUyOfzpbscdAFt/cy0Nxtw6hoAAAAA2yHoAAAAAB1syZIl6tatW9LH6NGj012eLXGNDgAAANDBfvCDH2jChAlJl7nd7k6u5tBA0AEAAAA6WHZ2trKzs9NdxiGFU9cAAAAA2A5BBwAAAIDtEHQAAAAA2A5BBwAAAIDtEHQAAAAA2A5BBwAAAOggQ4cO1YMPPpjuMg5JBB0AAAAAtkPQAQAAANBCKBSSaZrpLmOfEXQAAADQ6SzLkllXl5aHZVntqvGRRx7RwIEDWxzs/+AHP9BVV12lTZs26eyzz1ZeXp66deumY489Vv/4xz/2+T25//77NXbsWGVlZSk/P18//elPVVNTk9Dnvffe0ymnnKLMzEz16NFDZ5xxhvbs2SNJMk1T9957r4YPHy6v16vBgwfrnnvukST961//kmEY+vbbb2PbKi4ulmEY+uqrryRJTz31lLp3765XX31VRxxxhLxer7Zu3arVq1fr9NNPV+/evZWbm6tTTjlFH3/8cUJd3377rX70ox8pLy9PPp9PY8aM0auvvqra2lrl5OTof/7nfxL6/+1vf1NWVpaqq6v3+f3aG1eHbRkAAABohVVfry/GFaZl34d/vEZGZuZe+11wwQWaPXu23nrrLU2ePFmStGfPHr3xxhv629/+ppqaGk2bNk133323fD6f/vznP2v69On64osvNHjw4JTrcjgceuihhzR06FBt2bJFP/3pT/Xzn/9cixYtkhQOJpMnT9bVV1+thx56SC6XS2+99ZZCoZAkad68eXrsscf0wAMP6MQTT1Rpaak+//zzlGqoq6vT/Pnz9ac//Um9evVS3759tWXLFl111VV66KGHJEkLFizQtGnTtHHjRmVnZ8s0TU2dOlXV1dV69tlnddhhh2ndunVyOp3KysrSxRdfrCeffFI//OEPY/uJzmdnZ6f8PrUXQQcAAABIomfPnjrzzDP13HPPxYLOCy+8oJ49e2ry5MlyOp066qijYv3vvvtuvfzyy1q2bJlmzZqV8v7mzJkTmy4oKNB//ud/6ic/+Uks6Nx3330aP358bF6SRo8eLUmqrq7W73//ey1cuFBXXXWVJOmwww7TiSeemFINgUBAixYtSnhdkyZNSujzyCOPqEePHnr77bd11lln6R//+If+/e9/a/369Ro5cqQkadiwYbH+1157rSZOnKidO3dqwIABKi8v16uvvqqioqKUaksVQQcAAACdzsjI0OEfr0nbvtvrsssu049+9CMtWrRIXq9XS5Ys0cUXXyyn06na2lrdeeedevXVV7Vz504Fg0HV19erpKRkn+p666239Nvf/lbr1q1TVVWVgsGgGhoaVFtbq6ysLBUXF+uCCy5Iuu769evV2NgYC2T7yuPx6Mgjj0xoKysr0x133KE333xTX3/9tUKhkOrq6mKvs7i4WIMGDYqFnOa++93vavTo0Xr66ad122236ZlnntHgwYN18skn71ete0PQAQAAQKczDKNdp4+l2/Tp02Wapl577TUde+yxWrlype6//35J0i233KI33nhD//3f/63hw4crIyNDP/zhD+X3+1Pez9atWzVt2jTNnDlT//mf/6mePXvq3Xff1TXXXKNAICBJymgjoLW1TAqfFicp4fqk6Habb8cwjIS2GTNmaPfu3XrwwQc1ZMgQeb1eHX/88bHXubd9S+FRnYULF+q2227Tk08+qf/4j/9osZ8DjZsRAAAAAK3IyMjQeeedpyVLluj555/XyJEjVVgYvrZo5cqVmjFjhs4991yNHTtW/fr1i13Yn6qPPvpIwWBQCxYs0HHHHaeRI0dq586dCX2OPPJI/fOf/0y6/ogRI5SRkdHq8j59+kiSSktLY23FxcXtqm3lypWaPXu2pk2bptGjR8vr9aq8vDyhru3bt2vDhg2tbuPyyy9XSUmJHnroIX322Wex0+s6EkEHAAAAaMNll12m1157TU888YQuv/zyWPvw4cP10ksvqbi4WJ988okuvfTSfb4d82GHHaZgMKg//OEP2rx5s5555hk9/PDDCX3mzZun1atX66c//an+93//V59//rkWL16s8vJy+Xw+3Xrrrfr5z3+up59+Wps2bdIHH3ygxx9/PFZrfn6+fvOb32jDhg167bXXtGDBgnbVNnz4cD3zzDNav369PvzwQ1122WUJozinnHKKTj75ZJ1//vkqKirSli1b9Pe//12vv/56rE+PHj103nnn6ZZbbtGUKVM0aNCgfXqfUkHQAQAAANowadIk9ezZU1988YUuvfTSWPsDDzygHj16aOLEiZo+fbrOOOMMjRs3bp/2cfTRR+v+++/XvffeqzFjxmjJkiWaP39+Qp+RI0dqxYoV+uSTT/Td735Xxx9/vP7617/K5QpfjfKrX/1KN910k+644w6NGjVKF110kcrKyiRJbrdbzz//vD7//HMdddRRuvfee3X33Xe3q7YnnnhCe/bs0THHHKMrrrhCs2fPVt++fRP6vPjiizr22GN1ySWX6IgjjtDPf/7z2N3goq655hr5/X5dffXV+/Qepcqw2nsj8TSqqqpSbm6uKisrlZOTk+5yAAAAkIKGhgZt2bJFBQUF8vl86S4HabJkyRL97Gc/086dO+XxeNrs29bPTHuzATcjAAAAANBh6urqtGXLFs2fP18//vGP9xpyDhROXQMAAAA62JIlS9StW7ekj+h34djVfffdp6OPPlp5eXmaN29ep+2XU9cAAADQoTh1LfyFnl9//XXSZW63W0OGDOnkig5unLoGAAAAdAHZ2dnKzs5OdxmHFE5dAwAAQKfoAicS4SBxIH5WCDoAAADoUG63W1L4onSgPaI/K9GfnX3BqWsAAADoUE6nU927d499p0tmZqYMw0hzVTgYWZaluro6lZWVqXv37nI6nfu8LYIOAAAAOly/fv0kKRZ2gLZ079499jOzrwg6AAAA6HCGYah///7q27evAoFAusvBQcztdu/XSE4UQQcAAACdxul0HpCDWGBvuBkBAAAAANvZp6CzaNGi2Jf3FBYWauXKle1a77333pPL5dLRRx+9L7sFAAAAgHZJOegsXbpUc+bM0e233661a9fqpJNO0tSpU1VSUtLmepWVlbryyis1efLkfS4WAAAAANrDsFL8Np4JEyZo3LhxWrx4caxt1KhROuecczR//vxW17v44os1YsQIOZ1OvfLKKyouLm73PquqqpSbm6vKykrl5OSkUi4AAAAAG2lvNkhpRMfv92vNmjWaMmVKQvuUKVO0atWqVtd78skntWnTJv36179u134aGxtVVVWV8AAAAACA9kop6JSXlysUCikvLy+hPS8vT7t27Uq6zsaNG3XbbbdpyZIlcrnad5O3+fPnKzc3N/bIz89PpUwAAAAAh7h9uhlB82+ytSwr6bfbhkIhXXrppbrzzjs1cuTIdm9/3rx5qqysjD22bdu2L2UCAAAAOESl9D06vXv3ltPpbDF6U1ZW1mKUR5Kqq6v10Ucfae3atZo1a5YkyTRNWZYll8ulFStWaNKkSS3W83q98nq9qZQGAAAAADEpjeh4PB4VFhaqqKgoob2oqEgTJ05s0T8nJ0effvqpiouLY4+ZM2fq8MMPV3FxsSZMmLB/1QMAAABAEimN6EjS3LlzdcUVV2j8+PE6/vjj9eijj6qkpEQzZ86UFD7tbMeOHXr66aflcDg0ZsyYhPX79u0rn8/Xoh0AAAAADpSUg85FF12kiooK3XXXXSotLdWYMWO0fPlyDRkyRJJUWlq61+/UAQAAAICOlPL36KQD36MDAAAAQOqg79EBAAAAgK6AoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGyHoAMAAADAdgg6AAAAAGxnn4LOokWLVFBQIJ/Pp8LCQq1cubLVvi+99JJOP/109enTRzk5OTr++OP1xhtv7HPBAAAAALA3KQedpUuXas6cObr99tu1du1anXTSSZo6dapKSkqS9n/nnXd0+umna/ny5VqzZo2+973vafr06Vq7du1+Fw8AAAAAyRiWZVmprDBhwgSNGzdOixcvjrWNGjVK55xzjubPn9+ubYwePVoXXXSR7rjjjnb1r6qqUm5uriorK5WTk5NKuQAAAABspL3ZIKURHb/frzVr1mjKlCkJ7VOmTNGqVavatQ3TNFVdXa2ePXu22qexsVFVVVUJDwAAAABor5SCTnl5uUKhkPLy8hLa8/LytGvXrnZtY8GCBaqtrdWFF17Yap/58+crNzc39sjPz0+lTAAAAACHuH26GYFhGAnzlmW1aEvm+eef129+8xstXbpUffv2bbXfvHnzVFlZGXts27ZtX8oEAAAAcIhypdK5d+/ecjqdLUZvysrKWozyNLd06VJdc801euGFF3Taaae12dfr9crr9aZSGgAAAADEpDSi4/F4VFhYqKKiooT2oqIiTZw4sdX1nn/+ec2YMUPPPfecvv/97+9bpQAAAADQTimN6EjS3LlzdcUVV2j8+PE6/vjj9eijj6qkpEQzZ86UFD7tbMeOHXr66aclhUPOlVdeqd///vc67rjjYqNBGRkZys3NPYAvBQAAAADCUg46F110kSoqKnTXXXeptLRUY8aM0fLlyzVkyBBJUmlpacJ36jzyyCMKBoO6/vrrdf3118far7rqKj311FP7/woAAAAAoJmUv0cnHfgeHQAAAABSB32PDgAAAAB0BQQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0AEAAABgOwQdAAAAALZD0EnRG5/t0v/tqEx3GQAAAADa4Ep3AV1JTWNQ8176VN/U+nXm6H668fSROrxfdrrLAgAAANAMIzop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",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.DataFrame(history_6.history).plot(figsize=(10,7)) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "63926a7e-334b-4dd7-8ec2-d7d6572147e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_5\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d_7 (Conv2D)           (None, 26, 26, 32)        320       \n",
      "                                                                 \n",
      " max_pooling2d_7 (MaxPooling  (None, 13, 13, 32)       0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " conv2d_8 (Conv2D)           (None, 11, 11, 32)        9248      \n",
      "                                                                 \n",
      " max_pooling2d_8 (MaxPooling  (None, 5, 5, 32)         0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " flatten_5 (Flatten)         (None, 800)               0         \n",
      "                                                                 \n",
      " dense_12 (Dense)            (None, 128)               102528    \n",
      "                                                                 \n",
      " dense_13 (Dense)            (None, 128)               16512     \n",
      "                                                                 \n",
      " dense_14 (Dense)            (None, 10)                1290      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 129,898\n",
      "Trainable params: 129,898\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model_6.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f416563b-ce80-4552-a9f4-6b3436ed47c2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
