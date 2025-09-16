import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { api } from '../services/api'

interface User {
    id: number
    username: string
    email: string
    full_name?: string
    is_admin: boolean
}

interface AuthContextType {
    user: User | null
    token: string | null
    login: (username: string, password: string) => Promise<void>
    register: (username: string, email: string, password: string, full_name: string) => Promise<void>
    logout: () => void
    loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
    const context = useContext(AuthContext)
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider')
    }
    return context
}

interface AuthProviderProps {
    children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null)
    const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (token) {
            // Set token in API headers
            api.defaults.headers.common['Authorization'] = `Bearer ${token}`

            // Try to get user profile
            api.get('/users/profile')
                .then(response => {
                    setUser(response.data)
                })
                .catch(() => {
                    // Token is invalid, clear it
                    localStorage.removeItem('token')
                    setToken(null)
                    delete api.defaults.headers.common['Authorization']
                })
                .finally(() => {
                    setLoading(false)
                })
        } else {
            setLoading(false)
        }
    }, [token])

    const login = async (username: string, password: string) => {
        try {
            const response = await api.post('/auth/login', { username, password })
            const { access_token } = response.data

            localStorage.setItem('token', access_token)
            setToken(access_token)
            api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`

            // Get full user profile
            const userResponse = await api.get('/users/profile')
            setUser(userResponse.data)
        } catch (error: any) {
            throw new Error(error.response?.data?.detail || 'Login failed')
        }
    }

    const register = async (username: string, email: string, password: string, full_name: string) => {
        try {
            await api.post('/auth/register', {
                username,
                email,
                password,
                full_name
            })
            // Do not auto-login after registration; caller will handle navigation to login page
        } catch (error: any) {
            throw new Error(error.response?.data?.detail || 'Registration failed')
        }
    }

    const logout = () => {
        localStorage.removeItem('token')
        setToken(null)
        setUser(null)
        delete api.defaults.headers.common['Authorization']
    }

    const value = {
        user,
        token,
        login,
        register,
        logout,
        loading
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
} 