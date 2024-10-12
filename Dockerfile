# Usar una imagen base de Python
FROM python:3.12

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos e instalarlos
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido de tu proyecto al contenedor
COPY . .

# Exponer el puerto si tu aplicación lo necesita (opcional)
# EXPOSE 5000  # Descomenta si necesitas exponer un puerto

# Ejecutar el script principal
CMD ["python", "precios-de-viviendas.py"]