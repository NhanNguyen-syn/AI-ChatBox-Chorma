import React, { useEffect, useRef, useState } from 'react'
import { UploadCloud, Trash2, RefreshCcw, FileSpreadsheet } from 'lucide-react'
import { api } from '../services/api'
import toast from 'react-hot-toast'

interface DocItem {
  id: string
  filename: string
  file_type: string
  uploaded_by: string
  uploaded_at: string
  file_size: number
}

interface ExcelItem {
  id: string
  filename: string
  file_size: number
  uploaded_at: string
  sheet_names: string[]
  rows_processed: number
  collection_name: string
}

const formatBytes = (bytes: number) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const AdminDocuments: React.FC = () => {
  const [docs, setDocs] = useState<DocItem[]>([])
  const [excel, setExcel] = useState<ExcelItem[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadingExcel, setUploadingExcel] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const excelInputRef = useRef<HTMLInputElement>(null)

  const loadDocs = async () => {
    setLoading(true)
    try {
      const [docsRes, excelRes] = await Promise.all([
        api.get('/files/documents'),
        api.get('/admin/excel-files')
      ])
      setDocs(Array.isArray(docsRes.data) ? docsRes.data : (docsRes.data ?? []))
      setExcel(Array.isArray(excelRes.data) ? excelRes.data : (excelRes.data ?? []))
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Không thể tải danh sách dữ liệu')
      setDocs([])
      setExcel([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocs()
  }, [])

  // Chunked upload for large files (fast + reliable)
  const uploadViaChunks = async (file: File) => {
    const CHUNK_SIZE = 5 * 1024 * 1024 // 5MB
    const CONCURRENCY = 4
    const total = Math.ceil(file.size / CHUNK_SIZE)
    const uploadId = `${Date.now()}_${Math.random().toString(36).slice(2)}`

    const uploadChunk = (i: number) => {
      const start = i * CHUNK_SIZE
      const end = Math.min(start + CHUNK_SIZE, file.size)
      const blob = file.slice(start, end)
      const form = new FormData()
      form.append('file', blob, file.name)
      form.append('upload_id', uploadId)
      form.append('index', String(i))
      form.append('total', String(total))
      form.append('filename', file.name)
      return api.post('/files/upload-chunk', form, { headers: { 'Content-Type': 'multipart/form-data' } })
    }

    // Upload in batches with limited parallelism
    for (let i = 0; i < total; i += CONCURRENCY) {
      const batch: Promise<any>[] = []
      for (let j = i; j < Math.min(i + CONCURRENCY, total); j++) batch.push(uploadChunk(j))
      await Promise.all(batch)
    }

    const fin = new FormData()
    fin.append('upload_id', uploadId)
    fin.append('filename', file.name)
    fin.append('total', String(total))
    await api.post('/files/upload-chunk/finish', fin, { headers: { 'Content-Type': 'multipart/form-data' } })
  }

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return

    setUploading(true)
    let success = 0
    for (const file of Array.from(fileList)) {
      try {
        if (file.size > 15 * 1024 * 1024) {
          // Use chunked upload for files > 15MB
          await uploadViaChunks(file)
        } else {
          const form = new FormData()
          form.append('file', file)
          await api.post('/files/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
        }
        success += 1
      } catch (e: any) {
        toast.error(`${file.name}: ${e.response?.data?.detail || 'Upload thất bại'}`)
      }
    }
    if (success > 0) toast.success(`Tải lên thành công ${success} tệp`)
    await loadDocs()
    setUploading(false)
  }

  const handleExcelUpload = async (file: File | null) => {
    if (!file) return
    setUploadingExcel(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await api.post('/admin/upload-excel', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      toast.success(`Đã xử lý Excel: ${res.data?.filename}`)
      await loadDocs()
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Upload Excel thất bại')
    } finally {
      setUploadingExcel(false)
    }
  }

  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault()
    e.stopPropagation()
    handleFiles(e.dataTransfer.files)
  }

  const deleteDoc = async (id: string) => {
    if (!confirm('Xóa tài liệu này?')) return
    try {
      await api.delete(`/files/documents/${id}`)
      toast.success('Đã xóa tài liệu')
      setDocs((d) => d.filter((x) => x.id !== id))
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Không thể xóa')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Unstructured documents upload */}
      <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Quản lý dữ liệu</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">Upload tài liệu (PDF/DOC/DOCX/TXT/IMG) để lập chỉ mục và hỏi đáp</p>
          </div>
          <button
            onClick={loadDocs}
            className="btn-secondary inline-flex items-center gap-2"
          >
            <RefreshCcw className="h-4 w-4" /> Làm mới
          </button>
        </div>

        {/* Dropzone */}
        <div
          className="mt-6 rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 p-8 text-center hover:border-primary-400 dark:bg-[#0f0f0f] dark:border-gray-700"
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation() }}
          onDrop={onDrop}
        >
          <div className="flex flex-col items-center gap-3">
            <UploadCloud className="h-10 w-10 text-primary-600" />
            <p className="text-gray-700 dark:text-gray-300">Kéo thả tệp vào đây hoặc</p>
            <div>
              <button
                onClick={() => inputRef.current?.click()}
                className="btn-primary"
                disabled={uploading}
              >{uploading ? 'Đang tải...' : 'Chọn tệp'}</button>
              <input
                ref={inputRef}
                type="file"
                className="hidden"
                multiple
                accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
                onChange={(e) => handleFiles(e.target.files)}
              />
            </div>
            <p className="text-xs text-gray-500">Hỗ trợ: PDF, DOC, DOCX, TXT, PNG, JPG</p>
          </div>
        </div>
      </div>

      {/* Excel upload */}
      <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="h-6 w-6 text-green-600" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Upload Excel/CSV</h2>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Hệ thống sẽ làm sạch, chuẩn hóa và lập chỉ mục để có thể hỏi đáp bằng ngôn ngữ tự nhiên</p>
            </div>
          </div>
          <div>
            <button onClick={() => excelInputRef.current?.click()} className="btn-primary" disabled={uploadingExcel}>
              {uploadingExcel ? 'Đang xử lý...' : 'Chọn Excel/CSV'}
            </button>
            <input
              ref={excelInputRef}
              type="file"
              className="hidden"
              accept=".xlsx,.xls,.csv"
              onChange={(e) => handleExcelUpload(e.target.files?.[0] || null)}
            />
          </div>
        </div>
      </div>

      {/* List unstructured docs */}
      <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold dark:text-gray-100">Tài liệu đã tải lên</h2>
          {loading && <span className="text-sm text-gray-500">Đang tải...</span>}
        </div>
        {docs.length === 0 && !loading ? (
          <div className="text-gray-500 dark:text-gray-400">Chưa có tài liệu nào.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-gray-600 dark:text-gray-400 border-b dark:border-gray-700">
                  <th className="py-2 pr-4">Tên tệp</th>
                  <th className="py-2 pr-4">Loại</th>
                  <th className="py-2 pr-4">Dung lượng</th>
                  <th className="py-2 pr-4">Người upload</th>
                  <th className="py-2 pr-4">Thời gian</th>
                  <th className="py-2 pr-4 text-right">Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id} className="border-b last:border-0 dark:border-gray-700">
                    <td className="py-3 pr-4 font-medium text-gray-900 dark:text-gray-100">{d.filename}</td>
                    <td className="py-3 pr-4 uppercase dark:text-gray-300">{d.file_type}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{formatBytes(d.file_size)}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{d.uploaded_by}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{new Date(d.uploaded_at).toLocaleString('vi-VN')}</td>
                    <td className="py-3 pr-4">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => deleteDoc(d.id)}
                          className="inline-flex items-center gap-1 text-red-600 hover:text-red-700 px-2 py-1 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20"
                        >
                          <Trash2 className="h-4 w-4" /> Xóa
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Excel datasets list */}
      <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0d0d0d] dark:border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold dark:text-gray-100">Bộ dữ liệu Excel</h2>
        </div>
        {excel.length === 0 ? (
          <div className="text-gray-500 dark:text-gray-400">Chưa có dữ liệu Excel nào.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-gray-600 dark:text-gray-400 border-b dark:border-gray-700">
                  <th className="py-2 pr-4">Tên tệp</th>
                  <th className="py-2 pr-4">Sheets</th>
                  <th className="py-2 pr-4">Số dòng</th>
                  <th className="py-2 pr-4">Dung lượng</th>
                  <th className="py-2 pr-4">Thời gian</th>
                  <th className="py-2 pr-4">Collection</th>
                  <th className="py-2 pr-4 text-right">Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {excel.map((e) => (
                  <tr key={e.id} className="border-b last:border-0 dark:border-gray-700">
                    <td className="py-3 pr-4 font-medium text-gray-900 dark:text-gray-100">{e.filename}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{e.sheet_names.join(', ')}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{e.rows_processed}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{formatBytes(e.file_size)}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{new Date(e.uploaded_at).toLocaleString('vi-VN')}</td>
                    <td className="py-3 pr-4 dark:text-gray-300">{e.collection_name}</td>
                    <td className="py-3 pr-4">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={async () => { if (confirm('Xóa dữ liệu Excel này?')) { await api.delete(`/admin/excel-files/${e.id}`); toast.success('Đã xóa'); await loadDocs() } }}
                          className="inline-flex items-center gap-1 text-red-600 hover:text-red-700 px-2 py-1 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20"
                        >
                          <Trash2 className="h-4 w-4" /> Xóa
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default AdminDocuments

