package mock_framework.security;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 사내 표준 API 인가(Authorization) 어노테이션.
 * Controller의 메서드 상단에 선언하여 사내 시스템 접근 권한을 제어합니다.
 * * 사용 예시:
 * @CompanyAuth(role = "ADMIN", requireOtp = true)
 * public void updateUserInfo() { ... }
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface CompanyAuth {
    
    // 요구되는 사내 권한 등급 (기본값: "USER")
    String role() default "USER";

    // 2차 인증(OTP) 필수 여부 (망 분리 환경 적용 시 true)
    boolean requireOtp() default false;
}