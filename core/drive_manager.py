import os
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
import PyPDF2
from docx import Document
import mimetypes
import uuid
from datetime import datetime

# Настройки Google Drive API
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]

class DriveManager:
    def __init__(self):
        self.credentials = None
        self.service = None
        self.folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        self._authenticate()
    
    def _authenticate(self):
        """Аутентификация в Google Drive API."""
        try:
            # Используем тот же service account, что и для Google Sheets
            creds = Credentials.from_service_account_file(
                'service_account.json',
                scopes=SCOPES
            )
            self.service = build('drive', 'v3', credentials=creds)
            print("✅ Google Drive API подключен успешно")
        except Exception as e:
            print(f"❌ Ошибка подключения к Google Drive: {e}")
            self.service = None
    
    def create_folder(self, folder_name, parent_folder_id=None):
        """Создать папку в Google Drive."""
        if not self.service:
            return None
        
        try:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_folder_id:
                folder_metadata['parents'] = [parent_folder_id]
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            return folder.get('id')
        except HttpError as e:
            print(f"❌ Ошибка создания папки: {e}")
            return None
    
    def upload_file(self, file_path, file_name=None, folder_id=None, description=None):
        """Загрузить файл в Google Drive."""
        if not self.service:
            return None
        
        try:
            if not file_name:
                file_name = os.path.basename(file_path)
            
            # Определяем MIME тип
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            # Метаданные файла
            file_metadata = {
                'name': file_name,
                'description': description
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            # Загружаем файл
            with open(file_path, 'rb') as file:
                media = MediaIoBaseUpload(
                    file,
                    mimetype=mime_type,
                    resumable=True
                )
                
                file_obj = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id,name,webViewLink,size,createdTime'
                ).execute()
            
            return {
                'id': file_obj.get('id'),
                'name': file_obj.get('name'),
                'url': file_obj.get('webViewLink'),
                'size': file_obj.get('size'),
                'created_time': file_obj.get('createdTime')
            }
        except HttpError as e:
            print(f"❌ Ошибка загрузки файла: {e}")
            return None
    
    def upload_bytes(self, file_bytes, file_name, mime_type=None, folder_id=None, description=None):
        """Загрузить файл из байтов в Google Drive."""
        if not self.service:
            return None
        
        try:
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file_name)
                if mime_type is None:
                    mime_type = 'application/octet-stream'
            
            # Метаданные файла
            file_metadata = {
                'name': file_name,
                'description': description
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            # Загружаем файл из байтов
            media = MediaIoBaseUpload(
                io.BytesIO(file_bytes),
                mimetype=mime_type,
                resumable=True
            )
            
            file_obj = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink,size,createdTime'
            ).execute()
            
            return {
                'id': file_obj.get('id'),
                'name': file_obj.get('name'),
                'url': file_obj.get('webViewLink'),
                'size': file_obj.get('size'),
                'created_time': file_obj.get('createdTime')
            }
        except HttpError as e:
            print(f"❌ Ошибка загрузки файла из байтов: {e}")
            return None
    
    def get_file_content(self, file_id):
        """Получить содержимое файла."""
        if not self.service:
            return None
        
        try:
            file_obj = self.service.files().get(fileId=file_id).execute()
            mime_type = file_obj.get('mimeType', '')
            
            # Для Google Docs получаем текст
            if 'google-apps' in mime_type:
                return self._get_google_doc_content(file_id, mime_type)
            
            # Для обычных файлов скачиваем
            else:
                return self._download_file_content(file_id)
                
        except HttpError as e:
            print(f"❌ Ошибка получения содержимого файла: {e}")
            return None
    
    def _get_google_doc_content(self, file_id, mime_type):
        """Получить содержимое Google Doc."""
        try:
            if 'document' in mime_type:
                # Google Docs
                doc = self.service.files().export(
                    fileId=file_id,
                    mimeType='text/plain'
                ).execute()
                return doc.decode('utf-8')
            
            elif 'spreadsheet' in mime_type:
                # Google Sheets
                doc = self.service.files().export(
                    fileId=file_id,
                    mimeType='text/csv'
                ).execute()
                return doc.decode('utf-8')
            
            else:
                return None
        except Exception as e:
            print(f"❌ Ошибка получения содержимого Google Doc: {e}")
            return None
    
    def _download_file_content(self, file_id):
        """Скачать содержимое обычного файла."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = request.execute()
            return file_content
        except Exception as e:
            print(f"❌ Ошибка скачивания файла: {e}")
            return None
    
    def extract_text_from_file(self, file_id):
        """Извлечь текст из файла (PDF, DOCX, TXT)."""
        try:
            file_content = self.get_file_content(file_id)
            if not file_content:
                return None
            
            file_obj = self.service.files().get(fileId=file_id).execute()
            file_name = file_obj.get('name', '').lower()
            
            # PDF файлы
            if file_name.endswith('.pdf'):
                return self._extract_text_from_pdf(file_content)
            
            # DOCX файлы
            elif file_name.endswith('.docx'):
                return self._extract_text_from_docx(file_content)
            
            # TXT файлы
            elif file_name.endswith('.txt'):
                return file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
            
            # Google Docs
            elif 'google-apps.document' in file_obj.get('mimeType', ''):
                return file_content
            
            else:
                return None
                
        except Exception as e:
            print(f"❌ Ошибка извлечения текста: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_content):
        """Извлечь текст из PDF."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"❌ Ошибка извлечения текста из PDF: {e}")
            return None
    
    def _extract_text_from_docx(self, docx_content):
        """Извлечь текст из DOCX."""
        try:
            doc = Document(io.BytesIO(docx_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"❌ Ошибка извлечения текста из DOCX: {e}")
            return None
    
    def create_document_folder(self, document_type, date=None):
        """Создать папку для документа определенного типа."""
        if not self.service:
            return None
        
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        folder_name = f"{document_type}_{date}_{str(uuid.uuid4())[:8]}"
        return self.create_folder(folder_name, self.folder_id)
    
    def list_files_in_folder(self, folder_id=None):
        """Получить список файлов в папке."""
        if not self.service:
            return []
        
        try:
            query = f"'{folder_id or self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id,name,mimeType,createdTime,size,webViewLink)"
            ).execute()
            
            return results.get('files', [])
        except HttpError as e:
            print(f"❌ Ошибка получения списка файлов: {e}")
            return []
    
    def delete_file(self, file_id):
        """Удалить файл из Google Drive."""
        if not self.service:
            return False
        
        try:
            self.service.files().delete(fileId=file_id).execute()
            return True
        except HttpError as e:
            print(f"❌ Ошибка удаления файла: {e}")
            return False

# Глобальный экземпляр менеджера
drive_manager = DriveManager() 